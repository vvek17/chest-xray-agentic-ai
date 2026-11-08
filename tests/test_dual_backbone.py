from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from backend.dual_backbone import FusedHeads, class_prompt, prepare_images, scoring_targets


def test_class_prompt_keeps_fixed_order_and_token_collisions_use_sequence_scoring():
    classes = ["Covid-19", "Normal", "Pneumonia", "Tuberculosis"]
    assert ", ".join(classes) in class_prompt(classes)["user"]
    tokenizer = SimpleNamespace(encode=lambda c, **kw: {"A": [1, 2], "B": [1, 3]}[c], decode=str)
    assert scoring_targets(tokenizer, ["A", "B"])["mode"] == "full_sequence"
    tokenizer.encode = lambda c, **kw: [1]
    with pytest.raises(ValueError, match="distinct"):
        scoring_targets(tokenizer, ["A", "B"])


def test_scaler_uses_only_training_and_both_heads_receive_gradients():
    torch.manual_seed(42)
    train = torch.tensor([[1., 2., 3.], [3., 2., 7.]])
    heads = FusedHeads(3, 2)
    heads.fit_scaler(train)
    torch.testing.assert_close(heads.mean, torch.tensor([2., 2., 5.]))
    torch.testing.assert_close(heads.std, torch.tensor([1., 1e-6, 2.]))
    a, b = heads(train)
    y = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(a, y) + torch.nn.functional.binary_cross_entropy_with_logits(b, torch.eye(2))
    loss.backward()
    assert heads.softmax_head.weight.grad.abs().sum() > 0
    assert heads.sigmoid_head.weight.grad.abs().sum() > 0
    mean = heads.mean.clone()
    outputs = heads.probabilities(torch.tensor([[100., 2., -100.]]))
    expected = (outputs["softmax"] + outputs["sigmoid"]) / 2
    expected /= expected.sum(dim=1, keepdim=True)
    torch.testing.assert_close(outputs["averaged"], expected)
    torch.testing.assert_close(outputs["averaged"].sum(dim=1), torch.ones(1))
    torch.testing.assert_close(heads.mean, mean)


def test_cleaner_produces_one_shared_rgb_input_without_changing_original():
    image = Image.new("RGB", (32, 32), color=(20, 80, 140))
    original = image.tobytes()
    cleaned = prepare_images([image])[0]
    assert cleaned.mode == "RGB" and cleaned.getpixel((0, 0))[0] == cleaned.getpixel((0, 0))[1]
    assert image.tobytes() == original


def test_both_frozen_backbones_receive_same_cleaned_objects_and_raw_scores(monkeypatch):
    from backend import dual_backbone as module
    seen = {}
    calls = []
    original_cleaner = module.clean_xray
    def cleaner(image):
        calls.append(image)
        return original_cleaner(image)
    monkeypatch.setattr(module, "clean_xray", cleaner)
    class Inputs(dict):
        def to(self, *args, **kwargs):
            return self
    class SigProcessor:
        def __call__(self, images, **kwargs):
            seen["sig"] = images
            return Inputs(pixel_values=torch.ones(len(images), 1))
    class GemProcessor:
        tokenizer = SimpleNamespace(encode=lambda c, **kwargs: [int(c)], decode=lambda t: str(t))
        def apply_chat_template(self, messages, **kwargs):
            seen["gem"] = [m[1]["content"][0]["image"] for m in messages]
            assert kwargs["add_generation_prompt"]
            return Inputs(input_ids=torch.ones(len(messages), 3, dtype=torch.long),
                          attention_mask=torch.ones(len(messages), 3, dtype=torch.long))
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.dtype = torch.float32
        def get_image_features(self, pixel_values):
            return SimpleNamespace(pooler_output=torch.ones(len(pixel_values), 1152))
        def forward(self, input_ids, **kwargs):
            assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
            return SimpleNamespace(logits=torch.tensor([0., -1., -30., 2., 4., 5.]).repeat(len(input_ids), 1, 1))
    sig, gem = Backbone(), Backbone()
    extractor = module.FrozenFeatures({"model": sig, "processor": SigProcessor(), "dtype": torch.float32},
                                     {"model": gem, "processor": GemProcessor()}, ["0", "1", "2", "3"])
    features, raw = extractor.extract([Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))])
    assert len(calls) == 2
    assert all(a is b for a, b in zip(seen["sig"], seen["gem"]))
    assert not sig.training and not gem.training and not sig.weight.requires_grad and not gem.weight.requires_grad
    assert features.shape == (2, 1156)
    expected = torch.tensor([0., -1., -30., 2., 4., 5.]).log_softmax(dim=0)[:4]
    torch.testing.assert_close(torch.tensor(raw[0]["medgemma_raw_class_logprobs"]), expected)
    torch.testing.assert_close(features[0, 1152:], expected.clamp_min(-20))


def test_wilson_interval_known_example():
    from training.train_dual_backbone import wilson
    low, high = wilson(80, 100)
    assert low == pytest.approx(0.71117, abs=1e-5)
    assert high == pytest.approx(0.86663, abs=1e-5)
