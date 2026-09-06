"""Factual-recall probe for quantized checkpoints.

Why this exists
---------------
FORGE has now twice met its acceptance metric while the model was unusable for the task:

  * Qwen2.5-Coder hit its perplexity band and produced 0/8 syntactically valid Python.
  * Falcon-H1 produced 8/8 "coherent" English and then answered that a carrot cake is made
    of "carrot cake mix, carrot cake mix, carrot cake mix".

Both times the gate measured the wrong thing. Perplexity averages over a corpus and hides
targeted failure; a coherence score rewards fluent text regardless of whether it is true.
This probe asks short factual questions whose answers contain checkable keywords, so
world-knowledge damage shows up as a number instead of an anecdote.

It is deliberately crude -- keyword containment, not semantic grading. Crude and honest
beats sophisticated and unavailable.

    python bench/factual_probe.py out/model-a.gguf out/model-b.gguf
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# (prompt, {any-of keyword groups}) -- a probe passes when every group is hit.
PROBES: list[tuple[str, list[list[str]]]] = [
    ("Q: What are the main ingredients in a carrot cake? A:",
     [["carrot"], ["flour"], ["sugar"], ["egg"]]),
    ("Q: What is the capital of Australia? A:",
     [["canberra"]]),
    ("Q: How many legs does a spider have? A:",
     [["eight", "8"]]),
    ("Q: What gas do plants absorb during photosynthesis? A:",
     [["carbon dioxide", "co2"]]),
    ("Q: Who wrote the play Romeo and Juliet? A:",
     [["shakespeare"]]),
    ("Q: What is the chemical symbol for gold? A:",
     [["au"]]),
    ("Q: In which year did the Second World War end? A:",
     [["1945"]]),
    ("Q: What is the largest planet in the solar system? A:",
     [["jupiter"]]),
    ("Q: What is the boiling point of water at sea level in Celsius? A:",
     [["100"]]),
    ("Q: Which ocean lies between Africa and Australia? A:",
     [["indian"]]),
    ("Q: What organ pumps blood around the human body? A:",
     [["heart"]]),
    ("Q: What is the main ingredient in guacamole? A:",
     [["avocado"]]),
]


def generate(binary: pathlib.Path, model: pathlib.Path, prompt: str, ntokens: int) -> str:
    out = subprocess.run(
        [str(binary), "-m", str(model), "-n", str(ntokens), prompt],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    ).stdout
    out = out.replace("<|begin_of_text|>", "")
    idx = out.find(prompt)
    if idx >= 0:
        out = out[idx + len(prompt):]
    # Keep only the first answer; these models continue with further Q/A pairs.
    return out.split("Q:")[0].strip()


def is_degenerate(text: str, min_words: int = 8) -> bool:
    """A generation that just loops a phrase carries no information."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < min_words:
        return False
    tri = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return bool(tri) and len(set(tri)) / len(tri) < 0.5


def score(text: str, groups: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(k in low for k in group) for group in groups)


def evaluate(binary, model, ntokens, verbose):
    hits, degen, rows = 0, 0, []
    for prompt, groups in PROBES:
        text = generate(binary, model, prompt, ntokens)
        ok, bad = score(text, groups), is_degenerate(text)
        hits += ok
        degen += bad
        rows.append((prompt, text, ok, bad))
        if verbose:
            mark = "PASS" if ok else ("LOOP" if bad else "FAIL")
            print(f"  [{mark}] {prompt[3:60]:<58} {text[:70]!r}")
    return hits, degen, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--binary", default="llamacpp/llama.cpp/build/bin/llama-simple")
    ap.add_argument("--ntokens", type=int, default=40)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", help="write a markdown table here")
    args = ap.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    if not binary.exists():
        sys.exit(f"missing {binary}; build llama.cpp first")

    results = []
    for m in args.models:
        model = pathlib.Path(m).resolve()
        print(f"=== {model.name} ===")
        hits, degen, _ = evaluate(binary, model, args.ntokens, not args.quiet)
        n = len(PROBES)
        print(f"  factual recall {hits}/{n} ({hits/n:.0%})   degenerate {degen}/{n}\n")
        results.append({"model": model.name, "recall": f"{hits}/{n}",
                        "recall_pct": f"{hits/n:.0%}", "degenerate": f"{degen}/{n}"})

    if args.out:
        from forge.eval.report import write_report
        write_report(args.out, "Factual recall probe", results,
                     notes=f"{len(PROBES)} keyword-checked questions, "
                           f"{args.ntokens} tokens, greedy decoding via llama-simple.\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
