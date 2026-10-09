import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenomics.score_benchmark import (
    build_request, load_records, main, parse_questions, summarize, validate_config,
)
from tokenomics.plot_score_benchmark import plot_score_benchmark
import test_score_benchmark as ranking_tests

EXAMPLES = ranking_tests.EXAMPLES


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
                build_request(record, question_config(), 'd1')
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


class QuestionHTTPTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = ranking_tests.HTTPTests.asyncSetUp
    args = ranking_tests.HTTPTests.args
    run_quietly = ranking_tests.HTTPTests.run_quietly

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
