import argparse
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web

from tokenomics.score_benchmark import benchmark, load_workload, payloads, positive_ints, validate_response

WORKLOAD = {"query_prefix": [10], "query_suffix": [11], "filler_token_ids": [12, 13, 14],
            "items": [[20, 21], [22, 23, 24]], "label_token_ids": [30, 31]}


class WorkloadTests(unittest.TestCase):
    def test_states_match_across_concurrency_and_item_count(self):
        first = payloads(WORKLOAD, "test", (10, 1, 2), 0, 42)
        more = payloads(WORKLOAD, "test", (10, 2, 4), 0, 42)
        for a, b in zip(first, more):
            a, b = json.loads(a), json.loads(b)
            self.assertEqual(a["query"], b["query"])
            self.assertEqual(len(a["query"]), 10)
            self.assertEqual(a["items"], b["items"][:1])
        self.assertNotEqual(first, payloads(WORKLOAD, "test", (10, 1, 2), 1, 42))

    def test_invalid_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            for key, value in [("label_token_ids", [1, 1]), ("items", [[]]),
                               ("filler_token_ids", []), ("query_prefix", [True])]:
                with self.subTest(key=key):
                    path.write_text(json.dumps({**WORKLOAD, key: value}))
                    with self.assertRaises(ValueError):
                        load_workload(path)
        for text in ["0,1", "1,1", "-2"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_ints(text)

    def test_invalid_responses(self):
        for scores in [[], [[1]], [[float("nan"), 0]], [[-.1, 1.1]], [[.2, .2]]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                validate_response({"scores": scores, "usage": {"prompt_tokens": 10}}, 1, 2, 10)
        with self.assertRaises(ValueError):
            validate_response({"scores": [[.5, .5]], "usage": {"prompt_tokens": 9}}, 1, 2, 10)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workload = self.root / "workload.json"
        self.workload.write_text(json.dumps(WORKLOAD))
        self.mode = "mis"
        self.cache_disabled = True
        self.failure = None
        self.calls = self.active = self.peak = 0

        async def info(request):
            return web.json_response({"enable_mis": self.mode == "mis",
                                      "disable_radix_cache": self.cache_disabled})

        async def score(request):
            body = await request.json()
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(.002)
            self.active -= 1
            if self.failure == "http":
                return web.Response(status=500, text="injected failure")
            query, items = body["query"], body["items"]
            sequences = [query + item for item in items]
            tokens = sum(map(len, sequences))
            if self.mode == "mis":
                tokens = len(query) + sum(map(len, items)) + len(items) + 1
            if self.failure == "tokens":
                tokens += 1
            return web.json_response({"scores": [[.25, .75] for _ in items],
                                      "usage": {"prompt_tokens": tokens}})

        app = web.Application()
        app.router.add_get("/get_server_info", info)
        app.router.add_post("/v1/score", score)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    def args(self, name="results"):
        return SimpleNamespace(workload=str(self.workload), model="test", mode=self.mode,
                               api_base=self.url, query_lengths=[10], items=[1, 2],
                               batch_sizes=[1, 3], num_runs=2, warmup_runs=1, seed=42,
                               timeout=5, results_dir=str(self.root / name))

    async def test_sis_and_mis_accounting(self):
        for mode in ["sis", "mis"]:
            self.mode = mode
            args = self.args(mode)
            with contextlib.redirect_stdout(io.StringIO()):
                await benchmark(args)
            out = Path(args.results_dir)
            self.assertEqual(json.loads((out / "metadata.json").read_text())["status"], "complete")
            raw = [json.loads(line) for line in (out / "bursts.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw), 12)
            self.assertEqual(sum(row["warmup"] for row in raw), 4)
            for row in raw:
                n, c = row["items"], row["concurrency"]
                suffixes = 2 if n == 1 else 5
                processed = c * (10 + suffixes + n + 1) if mode == "mis" else c * (10 * n + suffixes)
                self.assertEqual(row["input_tokens"], processed)
                self.assertAlmostEqual(row["decisions_per_second"], c * n * 1000 / row["burst_latency_ms"])
                self.assertEqual(len(row["requests"]), c)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(summary), 4)
            self.assertTrue(all(row["runs"] == 2 for row in summary))
            self.assertTrue((out / "summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                await benchmark(args)
        self.assertEqual(self.peak, 3)

    async def test_failures_preserve_burst_without_retry_or_summary(self):
        for failure in ["http", "tokens"]:
            self.failure = failure
            before = self.calls
            args = self.args(failure)
            args.batch_sizes, args.items, args.warmup_runs = [3], [2], 0
            warning = io.StringIO()
            with contextlib.redirect_stderr(warning), self.assertRaises(RuntimeError):
                await benchmark(args)
            self.assertIn("WARNING: 3/3 requests failed", warning.getvalue())
            out = Path(args.results_dir)
            self.assertEqual(self.calls - before, 3)
            self.assertEqual(json.loads((out / "metadata.json").read_text())["status"], "failed")
            raw = [json.loads(line) for line in (out / "bursts.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw[0]["errors"], 3)
            self.assertNotIn("decisions_per_second", raw[0])
            self.assertFalse((out / "summary.json").exists())

    async def test_server_configuration_is_checked(self):
        args = self.args("mode")
        args.mode = "sis"
        with self.assertRaisesRegex(ValueError, "enable_mis"):
            await benchmark(args)
        self.cache_disabled = False
        with self.assertRaisesRegex(ValueError, "radix"):
            await benchmark(self.args("cache"))
        self.assertEqual(self.calls, 0)


class PlotTests(unittest.TestCase):
    def test_axes_include_all_item_counts_without_status_check(self):
        from matplotlib.figure import Figure
        from tokenomics.plot_score_benchmark import plot_score_benchmark
        from tokenomics.score_benchmark import METRICS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"status": "complete", "arguments": {"model": "test", "mode": "mis", "num_runs": 2}}
            (root / "metadata.json").write_text(json.dumps(metadata))
            rows = [{"query_tokens": 10, "items": n, "concurrency": 1,
                     **{metric: {"mean": value, "std": 0} for metric in METRICS}}
                    for n, value in [(1, 1), (16, 100)]]
            (root / "summary.json").write_text(json.dumps(rows))

            def inspect(figure, *args, **kwargs):
                self.assertEqual(len(figure.axes), 3)
                for ax in figure.axes:
                    self.assertEqual(len(ax.lines), 2)
                    self.assertGreaterEqual(ax.get_ylim()[1], 100)

            with patch.object(Figure, "savefig", autospec=True, side_effect=inspect):
                plot_score_benchmark(root, root / "plot.png")
            metadata["status"] = "failed"
            (root / "metadata.json").write_text(json.dumps(metadata))
            with patch.object(Figure, "savefig", autospec=True, side_effect=inspect):
                plot_score_benchmark(root, root / "plot.png")


if __name__ == "__main__":
    unittest.main()
