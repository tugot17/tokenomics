import asyncio
import contextlib
import io
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tokenomics import completion_benchmark as benchmark
from tokenomics.sampling import Scenario


class Sampler:
    def __init__(self):
        self.calls = []
        self.tokenizer = None

    def sample_batch(self, scenario, count):
        self.calls.append((scenario, count))
        return [SimpleNamespace(prompt=f'prompt-{random.random()}',
                                target_input_tokens=1024, target_output_tokens=256,
                                images=None) for _ in range(count)]


class WarmupTests(unittest.TestCase):
    def test_actual_workload_runs_concurrently_and_preserves_generation_options(self):
        sampler = Sampler()
        scenario = Scenario.from_string('D(1024,256)')
        active = peak = 0
        calls = []

        async def request(session, api, model, data, temperature, timeout, api_key, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            start = time.perf_counter()
            calls.append((dict(data), temperature, kwargs))
            await asyncio.sleep(0.002)
            active -= 1
            return dict(success=True, input_tokens=1024, output_tokens=data['max_tokens'],
                        ttft=0, ttfo=0, input_throughput=0, output_throughput=0,
                        output_latency=0.002, tpot=0, start_time=start,
                        end_time=time.perf_counter())

        loras = SimpleNamespace(assign_lora=lambda count: ['adapter'] * count)
        with patch.object(benchmark, 'single_request', request), contextlib.redirect_stdout(io.StringIO()):
            report = benchmark.warmup_workload(
                'http://local/v1', 'model', 0.7, 30, sampler, scenario, 4,
                num_warmup_runs=2, max_concurrency=4, max_tokens=128,
                n=2, stream=False, lora_config=loras, ignore_eos=True)
        self.assertEqual(peak, 4)
        self.assertEqual(len(calls), 8)
        self.assertEqual(sampler.calls, [(scenario, 4), (scenario, 4)])
        for data, temperature, options in calls:
            self.assertEqual(data['max_tokens'], 128)
            self.assertTrue(data['ignore_eos'])
            self.assertEqual(data['lora_name'], 'adapter')
            self.assertEqual(temperature, 0.7)
            self.assertEqual(options['n'], 2)
            self.assertFalse(options['stream'])
        self.assertEqual([r['output_tokens'] for r in report['runs']], [512, 512])
        self.assertTrue(all(r['failed'] == 0 for r in report['runs']))

    def test_failure_stops_warmup_and_zero_runs_sends_nothing(self):
        sampler = Sampler()

        async def failed(*args, **kwargs):
            return {'batch_total_seconds': 1, 'failures': {'total_requests': 2,
                    'successful': 1, 'failed': 1},
                    'tokens': {'input_per_request': [1024], 'output_per_request': [256]}}

        with patch.object(benchmark, 'run_batch_async', failed), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'point was not measured'):
                benchmark.warmup_workload('http://local', 'model', 0, 30,
                                          sampler, None, 2, num_warmup_runs=3)
        self.assertEqual(len(sampler.calls), 1)
        sampler.calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            report = benchmark.warmup_workload('http://local', 'model', 0, 30,
                                              sampler, None, 2, num_warmup_runs=0)
        self.assertEqual(report['runs'], [])
        self.assertEqual(sampler.calls, [])

    def test_image_payloads_use_disjoint_ids_and_preserve_real_images(self):
        samples = [SimpleNamespace(prompt='text', target_input_tokens=32,
                                   target_output_tokens=64, images=None)]
        with patch.object(benchmark, 'build_synthetic_image_uris',
                          side_effect=lambda size, count, start_id: [f'image-{start_id}']):
            warm, end = benchmark.build_user_requests(samples, 64, num_images=1,
                                                     start_image_id=2**31)
            measured, _ = benchmark.build_user_requests(samples, 64, num_images=1)
        self.assertNotEqual(warm[0]['images'], measured[0]['images'])
        self.assertNotEqual(warm[0]['prompt'], measured[0]['prompt'])
        self.assertEqual(end, 2**31 + 1)
        samples[0].images = ['real-image']
        real, _ = benchmark.build_user_requests(samples, 64, num_images=1)
        self.assertEqual(real[0]['images'], ['real-image'])
        self.assertEqual(real[0]['prompt'], 'text')

    def test_cli_keeps_measured_seeds_and_counts_separate_from_warmup(self):
        def execute(warmup_runs):
            sampler = Sampler()
            measured = []

            async def warm(requests, *args, **kwargs):
                n = len(requests)
                return {'batch_total_seconds': 123, 'failures': {'total_requests': n,
                        'successful': n, 'failed': 0},
                        'tokens': {'input_per_request': [1024] * n,
                                   'output_per_request': [256] * n}}

            def batch(requests, *args, **kwargs):
                measured.extend(requests)
                return {'tokens': {'output_per_request': [256] * len(requests)},
                        'batch_total_seconds': 2,
                        'prefill_metrics': {'ttft_per_request': []}}

            with tempfile.TemporaryDirectory() as directory:
                argv = ['completion', '--model', 'model', '--scenario', 'D(1024,256)',
                        '--max-concurrency', '4', '--num-runs', '1', '--max-tokens', '256',
                        '--warmup-runs', str(warmup_runs), '--results-dir', directory]
                with patch('sys.argv', argv), \
                     patch.object(benchmark, 'build_prompt_source', return_value=(
                         'scenario', sampler, Scenario.from_string('D(1024,256)'), None, None)), \
                     patch.object(benchmark, 'run_batch_async', warm), \
                     patch.object(benchmark, 'run_benchmark_batch', batch), \
                     patch.object(benchmark, 'calculate_stats', return_value={
                         'batch_metrics': {'e2e_tps': {'mean': 8192}}}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
                record = json.loads((Path(directory) / '4.json').read_text())
            return measured, record

        cold, cold_record = execute(0)
        warmed, warm_record = execute(2)
        self.assertEqual(cold, warmed)
        self.assertEqual(len(warmed), 64)
        self.assertEqual(warm_record['metadata']['warmup_mode'], 'workload')
        self.assertEqual(warm_record['warmup']['requests_per_run'], 4)
        self.assertEqual(len(warm_record['warmup']['runs']), 2)
        self.assertEqual(warm_record['result'], cold_record['result'])
        self.assertEqual(warm_record['warmup']['runs'][0]['wall_seconds'], 123)


if __name__ == '__main__':
    unittest.main()
