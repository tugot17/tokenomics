"""Run with the patched SGLang checkout on PYTHONPATH; no GPU required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin


class Harness(TokenizerManagerScoreMixin):
    is_generation = True
    tokenizer = None
    model_config = SimpleNamespace(is_multimodal=True)

    async def generate_request(self, request, raw_request):
        self.captured = request
        yield []

    def _process_single_item_scoring_results(self, *args, **kwargs):
        return 'ok'


class ImageRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_images_reach_each_sequence(self):
        harness = Harness()
        with patch('sglang.srt.managers.tokenizer_manager_score_mixin.get_exec', return_value=SimpleNamespace(features=SimpleNamespace(enable_mis=False))):
            result = await harness.score_request(query='<image> Query', items=[' A', ' B'], image_data=['image-one', 'image-two'], label_token_ids=[1, 2])
        self.assertEqual(result, 'ok')
        self.assertEqual(harness.captured.text, ['<image> Query A', '<image> Query B'])
        self.assertEqual(harness.captured.image_data, [['image-one', 'image-two'], ['image-one', 'image-two']])

    async def test_unsupported_modes_rejected_before_generation(self):
        for mis, multimodal, kwargs in [(True, True, {}), (False, False, {}), (False, True, {'query': [1], 'items': [[2]]}), (False, True, {'item_first': True})]:
            harness = Harness()
            harness.model_config = SimpleNamespace(is_multimodal=multimodal)
            request = dict(query='<image> Query', items=[' A'], image_data=['image'], label_token_ids=[1, 2])
            request.update(kwargs)
            with patch('sglang.srt.managers.tokenizer_manager_score_mixin.get_exec', return_value=SimpleNamespace(features=SimpleNamespace(enable_mis=mis))):
                with self.assertRaises(ValueError):
                    await harness.score_request(**request)
            self.assertFalse(hasattr(harness, 'captured'))


if __name__ == '__main__':
    unittest.main()
