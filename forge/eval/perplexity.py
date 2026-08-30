"""Perplexity on wikitext2, matching the convention used by the PTQ literature.

Non-overlapping windows of `seqlen` tokens, token-level mean NLL over the whole test
split, exponentiated once at the end. Reported numbers are comparable to GPTQ/QuaRot/
AWQ papers and to llama.cpp's `llama-perplexity` (which uses the same windowing).
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm


@torch.no_grad()
def perplexity(model, windows: torch.Tensor, device: torch.device, progress: bool = True) -> float:
    """windows: (n, seqlen) token ids from calib.datasets.evaluation_batch."""
    model.eval()
    total_nll = torch.zeros((), dtype=torch.float64)
    total_tokens = 0

    for i in tqdm(range(windows.shape[0]), disable=not progress, desc="ppl", leave=False):
        ids = windows[i : i + 1].to(device)
        logits = model(ids, use_cache=False).logits.float()

        # Standard causal shift: predict token t+1 from position t.
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = ids[:, 1:].reshape(-1)
        nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="sum")

        total_nll += nll.detach().cpu().double()
        total_tokens += shift_labels.numel()

    return float(torch.exp(total_nll / total_tokens))
