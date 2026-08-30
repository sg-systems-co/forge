"""Calibration data.

FORGE needs only 128-512 sequences to estimate the layer input second moments -- the whole
point of the project is that this is a calibration problem, not a pretraining one. We
follow the sampling convention GPTQ/QuaRot/AWQ all use so our numbers stay comparable:
concatenate the raw split, tokenize once, then draw fixed-length windows at random offsets.
"""

from __future__ import annotations

import torch

WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")
C4 = ("allenai/c4", "en")


def _concatenated_wikitext(tokenizer, split: str) -> torch.Tensor:
    from datasets import load_dataset

    data = load_dataset(*WIKITEXT2, split=split)
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids


def _c4_windows(tokenizer, nsamples: int, seqlen: int, seed: int) -> torch.Tensor:
    """C4 is streamed: draw documents until each one is long enough to yield a window."""
    from datasets import load_dataset

    stream = load_dataset(
        C4[0], C4[1], split="train", streaming=True,
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
    )
    gen = torch.Generator().manual_seed(seed)
    out = []
    for doc in stream:
        ids = tokenizer(doc["text"], return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = int(torch.randint(0, ids.shape[1] - seqlen, (1,), generator=gen))
        out.append(ids[:, start : start + seqlen])
        if len(out) == nsamples:
            break
    if len(out) < nsamples:
        raise RuntimeError(f"C4 stream exhausted after {len(out)} of {nsamples} samples")
    return torch.cat(out, dim=0)


def calibration_batch(
    tokenizer,
    dataset: str = "wikitext2",
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """Return (nsamples, seqlen) token ids for calibration.

    Deterministic given (dataset, nsamples, seqlen, seed) -- the seed is recorded in the
    GGUF so a checkpoint can be reproduced exactly.
    """
    if dataset == "c4":
        return _c4_windows(tokenizer, nsamples, seqlen, seed)
    if dataset != "wikitext2":
        raise ValueError(f"unknown calibration dataset {dataset!r}")

    ids = _concatenated_wikitext(tokenizer, "train")
    if ids.shape[1] <= seqlen:
        raise RuntimeError(f"calibration corpus has only {ids.shape[1]} tokens")

    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.shape[1] - seqlen, (nsamples,), generator=gen)
    return torch.cat([ids[:, s : s + seqlen] for s in starts.tolist()], dim=0)


def evaluation_batch(tokenizer, seqlen: int = 2048, split: str = "test") -> torch.Tensor:
    """Contiguous, non-overlapping windows of wikitext2 for perplexity.

    Deliberately *not* random: perplexity must be computed over the whole split in a fixed
    order so numbers are comparable across runs and against published baselines.
    """
    ids = _concatenated_wikitext(tokenizer, split)
    nwin = ids.shape[1] // seqlen
    return ids[0, : nwin * seqlen].reshape(nwin, seqlen)
