import argparse
import contextlib
import copy
import io
import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

from aiohttp import web
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenomics.plot_score_benchmark import plot_score_benchmark
from tokenomics.score_benchmark import (
    main, benchmark, build_request, load_records, parse_response, summarize,
    validate_config, validate_questions, parse_questions, resolve_label_ids,
)

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def test_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "Yes": 1, "No": 2, "A": 3, "B": 4, "C": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def config(name='pointwise'):
    cfg = dict(json.loads((EXAMPLES / f'score_{name}.json').read_text()), formulation=name)
    with patch('tokenizers.Tokenizer.from_pretrained', return_value=test_tokenizer()):
        resolve_label_ids(cfg, 'test-model')
    return cfg


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.record = load_records(EXAMPLES / 'data' / 'score_candidate_sets.jsonl')[0]

    def test_label_ids_come_from_tokenizer(self):
        cfg = dict(json.loads((EXAMPLES / 'score_pointwise.json').read_text()), formulation='pointwise')
        with tempfile.TemporaryDirectory() as tmp:
            test_tokenizer().save(str(Path(tmp) / 'tokenizer.json'))
            resolve_label_ids(cfg, tmp)
            self.assertEqual([label['token_id'] for label in cfg['labels']], [1, 2])
            cfg['labels'][0] = {'text': 'Yes No'}
            with self.assertRaises(ValueError):
                resolve_label_ids(cfg, tmp)
        cfg = config()
        with patch('tokenizers.Tokenizer.from_pretrained') as loader:
            resolve_label_ids(cfg, 'unused')
            loader.assert_not_called()

    def test_pointwise_isolation(self):
        original = build_request(self.record, config(), 'model')
        changed = copy.deepcopy(self.record)
        changed['candidates'][1] = 'UNIQUE REPLACEMENT'
        updated = build_request(changed, config(), 'model')
        self.assertEqual(original['query'], updated['query'])
        self.assertEqual(original['items'][0], updated['items'][0])
        self.assertEqual(original['items'][2], updated['items'][2])
        self.assertNotEqual(original['items'][1], updated['items'][1])
        for candidate in self.record['candidates']:
            self.assertNotIn(candidate, original['query'])

    def test_setwise_joint_choice(self):
        cfg = config('setwise')
        for count in (2, 3):
            record = dict(self.record, candidates=self.record['candidates'][:count])
            body = build_request(record, cfg, 'model')
            self.assertEqual(len(body['items']), 1)
            self.assertEqual(len(body['label_token_ids']), count)
            for candidate in record['candidates']:
                self.assertIn(candidate, body['items'][0])
        with self.assertRaises(ValueError):
            build_request(dict(self.record, candidates=['x'] * 4), cfg, 'model')

    def test_invalid_configs(self):
        invalid = [[], dict(config(), formulation='unknown'),
                   dict(config(), item_template='{options}'),
                   dict(config(), positive_label_index=True)]
        duplicate = config()
        duplicate['labels'][1]['token_id'] = duplicate['labels'][0]['token_id']
        invalid.append(duplicate)
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_config(cfg)

    def test_response_selection_and_validation(self):
        for name, scores in [('pointwise', [[.1, .9], [.8, .2], [.2, .8]]),
                             ('setwise', [[.1, .8, .1]])]:
            cfg = config(name)
            body = build_request(self.record, cfg, 'model')
            response = {'scores': scores, 'usage': {'prompt_tokens': 30}}
            self.assertEqual(parse_response(response, body, cfg, 3), (1, 30))
            invalid = [[], dict(response, usage=None), dict(response, scores=[]),
                       dict(response, scores=[[float('nan')] * len(row) for row in scores]),
                       dict(response, scores=[[.1] * len(row) for row in scores])]
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    parse_response(value, body, cfg, 3)

    def test_metrics_count_complete_sets_and_include_failure_time(self):
        success = dict(success=True, candidate_count=16, prompt_tokens=100, correct=True, latency_ms=10)
        result = summarize([{'wall_seconds': 2, 'requests': [success]},
                            {'wall_seconds': 3, 'requests': [{'success': False}]}])
        self.assertEqual(result['requests_per_second'], .2)
        self.assertEqual(result['completed_candidates'], 16)
        self.assertEqual(result['failed_requests'], 1)
        self.assertEqual(result['input_tokens_per_second'], 20)
        self.assertEqual(result['accuracy_on_successful_labelled_sets'], 1)

    def test_invalid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dataset.jsonl'
            for record in ([], {'state': 'x', 'candidates': ['a']},
                           {'state': 'x', 'candidates': ['a', 'b'], 'expected_index': 2}):
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_records(path)


class CLITests(unittest.TestCase):
    def setUp(self):
        loader = patch('tokenizers.Tokenizer.from_pretrained', return_value=test_tokenizer())
        loader.start()
        self.addCleanup(loader.stop)

    def test_formulation_is_required_and_validated(self):
        # Even an old config containing a formulation cannot replace the CLI flag.
        for extra in ([], ['--formulation', 'unknown']):
            argv = ['score', '--model', 'test', '--config', 'unused.json',
                    '--dataset', 'unused.jsonl', '--results-dir', 'unused', *extra]
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as exited:
                    main()
                self.assertEqual(exited.exception.code, 2)
                self.assertIn('--formulation', err.getvalue())

    def test_explicit_formulations_are_saved_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('pointwise', 'setwise'):
                out = Path(tmp) / mode
                argv = ['score', '--model', 'test', '--formulation', mode,
                        '--config', str(EXAMPLES / f'score_{mode}.json'),
                        '--dataset', str(EXAMPLES / 'data' / 'score_candidate_sets.jsonl'),
                        '--results-dir', str(out), '--dry-run']
                with patch('sys.argv', argv):
                    main()
                metadata = json.loads((out / 'metadata.json').read_text())
                self.assertEqual(metadata['formulation'], mode)
                self.assertEqual(metadata['config']['formulation'], mode)


class PlotTests(unittest.TestCase):
    def test_candidate_set_results_render(self):
        from tokenomics.plot_score_benchmark import plot_score_benchmark
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'metadata.json').write_text(json.dumps({
                'model': 'test', 'formulation': 'setwise'}))
            (path / '1.json').write_text(json.dumps({
                'batch_size': 1, 'decisions_per_second': 2,
                'input_tokens_per_second': 100, 'latency_ms_mean': 500,
                'failed_candidate_sets': 0}))
            (path / '2.json').write_text(json.dumps({
                'batch_size': 2, 'decisions_per_second': 0,
                'input_tokens_per_second': 0, 'latency_ms_mean': None,
                'failed_candidate_sets': 2}))
            output = path / 'plot.png'
            plot_score_benchmark(path, output)
            self.assertTrue(output.read_bytes().startswith(b'\x89PNG'))


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader = patch('tokenizers.Tokenizer.from_pretrained', return_value=test_tokenizer())
        loader.start()
        self.addCleanup(loader.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.posts = []
        self.mis = False
        self.info_status = 200
        self.malformed = False

        async def info(request):
            return web.json_response({'enable_mis': self.mis}, status=self.info_status)

        async def score(request):
            body = await request.json()
            self.posts.append(body)
            if self.malformed:
                return web.json_response({'scores': []})
            n = len(body['label_token_ids'])
            return web.json_response({'scores': [[1 / n] * n for _ in body['items']],
                                      'usage': {'prompt_tokens': 100}})

        app = web.Application()
        app.router.add_get('/get_server_info', info)
        app.router.add_post('/v1/score', score)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        self.base = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1'

    def args(self, name, **overrides):
        values = dict(formulation=name, config=str(EXAMPLES / f'score_{name}.json'),
                      dataset=str(EXAMPLES / 'data' / 'score_candidate_sets.jsonl'), model='test-model',
                      results_dir=str(Path(self.tmp.name) / name), num_prompts=None,
                      batch_sizes=[1, 2], num_runs=2, warmup_runs=1, seed=42,
                      api_base=self.base, api_key=None, timeout=5, dry_run=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    async def run_quietly(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return await benchmark(args)

    async def test_all_presets_replay_same_records(self):
        digests = []
        request_hashes = {}
        for name, self.mis in (('pointwise', False), ('pointwise', True), ('setwise', False), ('setwise', True)):
            self.posts.clear()
            args = self.args(name, results_dir=str(Path(self.tmp.name) / f'{name}_{self.mis}'))
            self.assertTrue(await self.run_quietly(args))
            out = Path(args.results_dir)
            metadata = json.loads((out / 'metadata.json').read_text())
            self.assertEqual(metadata['formulation'], name)
            digests.append(metadata['dataset_sha256'])
            self.assertEqual(metadata['server']['enable_mis'], self.mis)
            if name in request_hashes:
                self.assertEqual(request_hashes[name], metadata['request_sha256'])
            request_hashes[name] = metadata['request_sha256']
            self.assertEqual(len(self.posts), 18)  # (warmup + two runs) * three sets * two concurrency levels
            self.assertTrue(all(len(b['items']) == (1 if name.startswith('setwise') else 3) for b in self.posts))
            for batch in (1, 2):
                result = json.loads((out / f'{batch}.json').read_text())
                self.assertEqual(result['formulation'], name)
                self.assertEqual(result['completed_requests'], 6)
                self.assertEqual(result['completed_candidates'], 18)
                self.assertGreater(result['latency_ms_mean'], 0)
            self.assertEqual(len((out / 'bursts.jsonl').read_text().splitlines()), 10)
        self.assertEqual(len(set(digests)), 1)

    async def test_server_metadata_is_optional(self):
        self.info_status = 404
        args = self.args('pointwise')
        self.assertTrue(await self.run_quietly(args))
        metadata = json.loads((Path(args.results_dir) / 'metadata.json').read_text())
        self.assertIn('server_info_error', metadata)
        self.assertTrue(self.posts)

    async def test_measured_failures_are_preserved(self):
        self.malformed = True
        args = self.args('pointwise', warmup_runs=0)
        self.assertFalse(await self.run_quietly(args))
        result = json.loads((Path(args.results_dir) / '1.json').read_text())
        self.assertEqual(result['failed_requests'], 6)
        self.assertEqual(result['requests_per_second'], 0)
        self.assertIsNone(result['latency_ms_mean'])

    async def test_warmup_failure_aborts(self):
        self.malformed = True
        args = self.args('setwise')
        with self.assertRaises(RuntimeError):
            await self.run_quietly(args)
        self.assertTrue((Path(args.results_dir) / 'warmup_failure.json').exists())
        self.assertFalse((Path(args.results_dir) / '1.json').exists())

    async def test_dry_run_and_overwrite_guard(self):
        args = self.args('setwise', dry_run=True, api_base='http://invalid.invalid')
        self.assertTrue(await self.run_quietly(args))
        self.assertEqual(self.posts, [])
        self.assertEqual(len((Path(args.results_dir) / 'requests.jsonl').read_text().splitlines()), 3)
        with self.assertRaises(FileExistsError):
            await self.run_quietly(args)

    async def test_questions_sis_mis_replay_and_plot(self):
        hashes = []
        for self.mis in (False, True):
            self.posts.clear()
            args = self.args('pointwise', workload='questions', formulation=None,
                             config=str(EXAMPLES / 'score_questions_d1.json'),
                             dataset=str(EXAMPLES / 'data/score_questions.jsonl'),
                             results_dir=str(Path(self.tmp.name) / str(self.mis)), num_runs=1)
            self.assertTrue(await self.run_quietly(args))
            out = Path(args.results_dir)
            hashes.append(json.loads((out / 'metadata.json').read_text())['request_sha256'])
            self.assertTrue(all(not b['apply_softmax'] for b in self.posts))
            result = json.loads((out / '1.json').read_text())
            self.assertEqual(result['completed_questions'], 2)
            self.assertEqual(result['completed_requests'], 1)
            self.assertEqual(result['questions_per_second'], 2 * result['requests_per_second'])
            self.assertEqual(result['accuracy_on_successful_labelled_questions'], 1)
            plot_score_benchmark(out, out / 'plot.png')
            self.assertTrue((out / 'plot.png').is_file())

            self.malformed = True
            args.results_dir += '-failed'
            args.warmup_runs = 0
            self.assertFalse(await self.run_quietly(args))
            failed = json.loads((Path(args.results_dir) / '1.json').read_text())
            self.assertEqual(failed['questions_per_second'], 0)
            self.assertEqual(failed['failed_requests'], 1)
            self.assertIsNone(failed['accuracy_on_successful_labelled_questions'])
            self.malformed = False
        self.assertEqual(hashes[0], hashes[1])


def question_config():
    return dict(json.loads((EXAMPLES / 'score_questions_d1.json').read_text()), workload='questions', formulation=None)


def question_record():
    return load_records(EXAMPLES / 'data/score_questions.jsonl', 'questions')[0]


class QuestionTests(unittest.TestCase):
    def test_exact_prompt_union_and_isolation(self):
        record = question_record()
        body = build_request(record, question_config(), 'd1')
        self.assertEqual(body['query'], '<|startoftext|><|im_start|>user\nI was charged twice this month, please refund one of them.\n\n\nQUESTION:\n')
        self.assertEqual(body['items'][0], 'Is the customer asking for a refund?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n')
        self.assertEqual(body['items'][1], 'Which team should handle this?\n\nOptions:\nA Charges, refunds, invoices\nB App or site faults\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n')
        self.assertEqual(body['label_token_ids'], [11683, 12447, 2243, 4547, 19598, 41, 334, 42, 378])
        self.assertFalse(body['apply_softmax'])
        record['questions'][1]['prompt'] = 'Different question'
        changed = build_request(record, question_config(), 'd1')
        self.assertEqual(body['query'], changed['query'])
        self.assertEqual(body['items'][0], changed['items'][0])

    def test_max_pool_then_normalize_per_question(self):
        record = question_record()
        body = build_request(record, question_config(), 'd1')
        # Sum pooling would choose no in row one. Irrelevant labels must not
        # influence its denominator; alias count must not influence the answer.
        rows = [[.1, .2, .15, .15, .15, .01, .01, .01, .01],
                [.1, .1, .1, .1, .1, .02, .03, .01, .09]]
        response = {'scores': rows, 'usage': {'prompt_tokens': 123}}
        answers, tokens = parse_questions(response, body, record)
        self.assertEqual(tokens, 123)
        self.assertEqual([a['selected_index'] for a in answers], [0, 1])
        self.assertEqual([a['correct'] for a in answers], [True, False])
        self.assertAlmostEqual(answers[0]['probabilities'][0], .2 / .35)
        self.assertEqual(answers[1]['probabilities'], [.25, .75])
        for bad in ([], [[0.] * 9] * 2, [[float('nan')] * 9] * 2, [[-1.] * 9] * 2, [[.1] * 8] * 2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_questions(dict(response, scores=bad), body, record)

    def test_variable_counts_reused_labels_and_unlabelled_questions(self):
        for count in (1, 8, 32):
            record = question_record()
            question = record['questions'][0]
            question.pop('expected_index')
            record['questions'] = [copy.deepcopy(question) for _ in range(count)]
            body = build_request(record, question_config(), 'd1')
            self.assertEqual(len(body['items']), count)
            self.assertEqual(len(body['label_token_ids']), 5)
            answers, _ = parse_questions({'scores': [[.1] * 5] * count, 'usage': {'prompt_tokens': 10}}, body, record)
            self.assertEqual(len(answers), count)
            self.assertTrue(all(a['correct'] is None for a in answers))

    def test_validation(self):
        for mutate in (
            lambda r: r.update(questions=[]),
            lambda r: r['questions'][0].update(prompt=None),
            lambda r: r['questions'][0].update(expected_index=2),
            lambda r: r['questions'][0]['labels'][0].update(token_ids=[]),
            lambda r: r['questions'][0]['labels'][0].update(token_ids=[True]),
            lambda r: r['questions'][0]['labels'][0].update(token_ids=[2243]),
        ):
            record = question_record()
            mutate(record)
            with self.assertRaises(ValueError):
                validate_questions(record)
        with self.assertRaises(ValueError):
            validate_config(dict(question_config(), formulation='setwise'))
        with self.assertRaises(ValueError):
            validate_config(dict(question_config(), item_template='{candidate}'))

    def test_metrics_weight_questions_not_requests_and_include_failures(self):
        def request(answers):
            return dict(success=True, answers=[{'correct': a} for a in answers],
                        question_count=len(answers), prompt_tokens=100, latency_ms=20)
        result = summarize([{'wall_seconds': 2, 'requests': [request([True]), request([True, False, None])]},
                            {'wall_seconds': 3, 'requests': [{'success': False, 'question_count': 8}]}], 'questions')
        self.assertEqual(result['requests_per_second'], 2 / 5)
        self.assertEqual(result['questions_per_second'], 4 / 5)
        self.assertEqual(result['attempted_questions'], 12)
        self.assertEqual(result['failed_requests'], 1)
        self.assertEqual(result['input_tokens_per_second'], 40)
        self.assertEqual(result['accuracy_on_successful_labelled_questions'], 2 / 3)
        self.assertNotIn('completed_candidates', result)

    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            argv = ['score', '--model', 'd1', '--workload', 'questions',
                    '--config', str(EXAMPLES / 'score_questions_d1.json'),
                    '--dataset', str(EXAMPLES / 'data/score_questions.jsonl'),
                    '--results-dir', str(out), '--dry-run']
            with patch('sys.argv', argv):
                main()
            metadata = json.loads((out / 'metadata.json').read_text())
            self.assertEqual(metadata['workload'], 'questions')
            self.assertIsNone(metadata['formulation'])
            self.assertNotIn('candidate_sets', metadata)


if __name__ == '__main__':
    unittest.main()
