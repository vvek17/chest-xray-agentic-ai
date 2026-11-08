"""Frozen first-token scoring, with one shared MedGemma prefill for score and text."""

import torch

from backend.medgemma import ask_medgemma
from backend.zero_shot_ensemble import parse_label

PROMPT = {
    "system": "You are an expert radiologist.",
    "user": ("Look at this chest X-ray. Is it Normal or Pneumonia? "
             "Answer with exactly Normal or Pneumonia and nothing else."),
    "max_tokens": 10,
}


def label_token_ids(tokenizer):
    sequences = {label: tokenizer.encode(label, add_special_tokens=False)
                 for label in ["Normal", "Pneumonia"]}
    assert all(sequences.values()), "Empty label tokenization"
    ids = {label: tokens[0] for label, tokens in sequences.items()}
    assert ids["Normal"] != ids["Pneumonia"], "Label first tokens must differ"
    return {"first_token_ids": ids, "full_token_sequences": sequences,
            "decoded_first_tokens": {label: tokenizer.decode([token]) for label, token in ids.items()},
            "tokenization": "Exact label strings, add_special_tokens=False; first token only"}


def first_token_score(logits, ids):
    """Normalize across the complete vocabulary before subtracting label log probabilities."""
    vector = logits[0, -1, :].detach().float().cpu()
    assert vector.ndim == 1 and 0 <= min(ids.values()) and max(ids.values()) < len(vector)
    log_probs = torch.log_softmax(vector, dim=-1)
    normal, pneumonia = ids["Normal"], ids["Pneumonia"]
    return {
        "medgemma_score": (log_probs[pneumonia] - log_probs[normal]).item(),
        "normal_first_token_logit": vector[normal].item(),
        "pneumonia_first_token_logit": vector[pneumonia].item(),
        "normal_first_token_logprob": log_probs[normal].item(),
        "pneumonia_first_token_logprob": log_probs[pneumonia].item(),
        "raw_argmax_token_id": int(vector.argmax().item()),
        "nonfinite_vocab_logits": int((~torch.isfinite(vector)).sum().item()),
        "logits_shape": list(logits.shape),
        "vocab_size": len(vector),
    }


def medgemma_score_and_answer(medgemma, cleaned_image, tokens):
    """Capture raw last-position logits from the first forward inside ask_medgemma.

    The first generate forward is the sole image/prompt prefill. Later forwards
    decode the short greedy answer using the same KV cache. The hook observes
    unprocessed logits, before any generation logit processors; no second prefill.
    """
    captured = []

    def capture_first_forward(module, args, output):
        if not captured:
            captured.append(first_token_score(output.logits, tokens["first_token_ids"]))

    handle = medgemma["model"].register_forward_hook(capture_first_forward)
    try:
        raw = ask_medgemma(medgemma, cleaned_image, PROMPT)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("First forward was not observed; refusing an unverified score")
    result = captured[0]
    result["medgemma_raw_answer"] = raw
    result["medgemma_greedy_answer"] = parse_label(raw)
    result["raw_argmax_token_text"] = medgemma["processor"].tokenizer.decode(
        [result["raw_argmax_token_id"]], skip_special_tokens=False)
    return result
