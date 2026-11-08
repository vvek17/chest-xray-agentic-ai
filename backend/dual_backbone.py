"""Frozen MedSigLIP + MedGemma features and two jointly trained linear heads.

Separate from the legacy combined probe. No original models are overwritten.
"""

import copy
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import MEDGEMMA_NAME, load_medgemma
from backend.zero_shot_ensemble import MODEL_NAME, load_zero_shot


def class_prompt(classes):
    return {"system": "You are an expert radiologist.",
            "user": "Look at this chest X-ray. Which one is it: " + ", ".join(classes)
            + "? Answer with exactly one of these names and nothing else."}


def scoring_targets(tokenizer, classes):
    sequences = [tokenizer.encode(c, add_special_tokens=False) for c in classes]
    if not all(sequences) or len({tuple(s) for s in sequences}) != len(classes):
        raise ValueError("Class token sequences must be non-empty and distinct")
    first = [s[0] for s in sequences]
    return {"classes": list(classes), "sequences": sequences, "first_token_ids": first,
            "mode": "first_token" if len(set(first)) == len(first) else "full_sequence",
            "decoded_first_tokens": [tokenizer.decode([t]) for t in first]}


def prepare_images(images):
    """One shared cleaning operation per input, used by both frozen backbones."""
    cleaned = []
    for image in images:
        if isinstance(image, (str, Path)):
            with Image.open(image) as original:
                cleaned.append(clean_xray(original))
        else:
            cleaned.append(clean_xray(image))
    if any(image.mode != "RGB" for image in cleaned):
        raise ValueError("Shared cleaning must return RGB images")
    return cleaned


def pixel_fingerprint(image):
    return hashlib.sha256(str(image.size).encode() + image.mode.encode() + image.tobytes()).hexdigest()


class FrozenFeatures:
    def __init__(self, siglip, medgemma, classes):
        self.siglip, self.medgemma = siglip, medgemma
        self.classes = tuple(classes)
        self.prompt = class_prompt(classes)
        self.targets = scoring_targets(medgemma["processor"].tokenizer, classes)
        for parts in (siglip, medgemma):
            parts["model"].eval().requires_grad_(False)
        self.device = next(medgemma["model"].parameters()).device

    @classmethod
    def load(cls, classes):
        return cls(load_zero_shot(), load_medgemma(), classes)

    def metadata(self):
        return {"models": [MODEL_NAME, MEDGEMMA_NAME], "classes": list(self.classes),
                "prompt": self.prompt, "targets": self.targets,
                "device": str(self.device),
                "dtypes": [str(self.siglip["dtype"]), str(self.medgemma["model"].dtype)],
                "processors": {name: parts["processor"].image_processor.to_dict()
                               for name, parts in (("medsiglip", self.siglip), ("medgemma", self.medgemma))},
                "shared_cleaner": "backend.clean_image.clean_xray; exactly once per image before either processor",
                "medsiglip_features": "1152-dimensional L2-normalized image embedding",
                "medgemma_features": "Full-vocabulary log-softmax; fixed class order; clip minimum -20 before train-only scaling",
                "model_revisions": [getattr(parts["model"].config, "_commit_hash", None)
                                    for parts in (self.siglip, self.medgemma)]}

    def _messages(self, image):
        return [{"role": "system", "content": [{"type": "text", "text": self.prompt["system"]}]},
                {"role": "user", "content": [{"type": "image", "image": image},
                                               {"type": "text", "text": self.prompt["user"]}]}]

    @torch.inference_mode()
    def extract(self, images):
        cleaned = prepare_images(images)
        fingerprints = [pixel_fingerprint(image) for image in cleaned]
        # The same PIL objects go to both processors; no model-specific cleaning.
        sig_inputs = self.siglip["processor"](images=cleaned, return_tensors="pt")
        sig_inputs = sig_inputs.to(self.device, self.siglip["dtype"])
        sig = self.siglip["model"].get_image_features(**sig_inputs).pooler_output.float().cpu()
        if sig.shape != (len(cleaned), 1152) or not torch.isfinite(sig).all() or (sig.norm(dim=1) == 0).any():
            raise ValueError("Invalid MedSigLIP image embeddings")
        sig = nn.functional.normalize(sig, dim=1)
        processor, model = self.medgemma["processor"], self.medgemma["model"]
        inputs = processor.apply_chat_template([self._messages(image) for image in cleaned],
                                               add_generation_prompt=True, tokenize=True, padding=True,
                                               return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=model.dtype)
        # One fixed prompt and resized image per item must have equal lengths.
        # This makes the last batch position the first-answer prediction for all.
        if not bool(inputs["attention_mask"][:, -1].all()):
            raise ValueError("Right-padded prompts: refusing scores from padding positions")
        if len(set(inputs["attention_mask"].sum(dim=1).tolist())) != 1:
            raise ValueError("Unexpected variable prompt lengths")
        sequence_mode = self.targets["mode"] == "full_sequence"
        output = model(**inputs, use_cache=sequence_mode, logits_to_keep=1)
        logits = output.logits[:, -1, :].float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite MedGemma vocabulary logits")
        logprobs = logits.log_softmax(dim=-1)
        first_ids = self.targets["first_token_ids"]
        scores = logprobs[:, first_ids].clone()
        if sequence_mode:
            # Reuse the sole image prefill and an independent KV-cache per name.
            # The fixed four-class dataset currently has distinct first tokens.
            for k, sequence in enumerate(self.targets["sequences"]):
                cache = copy.deepcopy(output.past_key_values)
                mask = inputs["attention_mask"]
                for previous, target in zip(sequence, sequence[1:]):
                    mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
                    token = torch.full((len(cleaned), 1), previous, dtype=torch.long, device=self.device)
                    continuation = model(input_ids=token, attention_mask=mask,
                                         token_type_ids=torch.zeros_like(token), past_key_values=cache,
                                         use_cache=True, logits_to_keep=1)
                    cache = continuation.past_key_values
                    next_logits = continuation.logits[:, -1, :].float().cpu()
                    if not torch.isfinite(next_logits).all():
                        raise ValueError("Non-finite MedGemma sequence logits")
                    scores[:, k] += next_logits.log_softmax(dim=-1)[:, target]
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite class scores")
        assert fingerprints == [pixel_fingerprint(image) for image in cleaned], "Processor mutated shared cleaned image"
        features = torch.cat([sig, scores.clamp_min(-20)], dim=1)
        raw = []
        for i in range(len(cleaned)):
            token = int(logits[i].argmax())
            raw.append({"medgemma_raw_class_logprobs": scores[i].tolist(),
                        "medgemma_first_token_class_logits": logits[i, first_ids].tolist(),
                        "medgemma_argmax_token_id": token,
                        "medgemma_argmax_token_text": processor.tokenizer.decode([token]),
                        "cleaned_image_sha256": fingerprints[i], "cleaned_size": list(cleaned[i].size)})
        return features, raw


class FusedHeads(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))
        self.softmax_head = nn.Linear(features, classes)
        self.sigmoid_head = nn.Linear(features, classes)

    def fit_scaler(self, train_features):
        if not torch.isfinite(train_features).all():
            raise ValueError("Non-finite training features")
        self.mean.copy_(train_features.mean(dim=0))
        self.std.copy_(train_features.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, features):
        standardized = (features - self.mean) / self.std
        return self.softmax_head(standardized), self.sigmoid_head(standardized)

    def probabilities(self, features):
        soft_logits, sig_logits = self(features)
        soft, sig = soft_logits.softmax(dim=-1), sig_logits.sigmoid()
        averaged = (soft + sig) / 2
        final = averaged / averaged.sum(dim=-1, keepdim=True)
        if not torch.isfinite(final).all():
            raise ValueError("Non-finite fused probabilities")
        return {"softmax": soft, "sigmoid": sig, "averaged": final}


class DualBackboneClassifier:
    """One predict call: shared clean -> both frozen models -> both learned heads."""
    def __init__(self, extractor, heads, classes):
        self.extractor, self.heads, self.classes = extractor, heads.eval(), tuple(classes)

    @classmethod
    def load(cls, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        extractor = FrozenFeatures.load(saved["classes"])
        if extractor.metadata() != saved["extractor_metadata"]:
            # JSON converts tuples to lists in metadata stored by the runner.
            import json
            assert json.loads(json.dumps(extractor.metadata())) == saved["extractor_metadata"], "Extractor changed"
        heads = FusedHeads(saved["feature_dim"], len(saved["classes"]))
        heads.load_state_dict(saved["state_dict"])
        return cls(extractor, heads, saved["classes"])

    @torch.inference_mode()
    def predict(self, image):
        features, _ = self.extractor.extract([image])
        probabilities = self.heads.probabilities(features)["averaged"][0]
        return dict(sorted(zip(self.classes, probabilities.tolist()), key=lambda item: item[1], reverse=True))
