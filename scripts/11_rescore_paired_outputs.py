#!/usr/bin/env python3
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def option_values(row):
    out = {}
    for letter in "ABCD":
        if letter in row and str(row[letter]).lower() != "nan":
            out[letter] = str(row[letter]).strip()
    return out


def parse_letter_prefix(candidate):
    m = re.match(r"\s*([A-D])(?:\s*[\).,:;\-]|\s+|$)", candidate, flags=re.I)
    return m.group(1).upper() if m else None


def exact_semantic_match(candidate, options):
    cand = normalize(candidate)
    if not cand:
        return None

    matches = []
    for letter, value in options.items():
        val = normalize(value)
        if not val:
            continue
        # Exact answer text or answer text embedded in a short final-answer phrase.
        if cand == val or re.search(rf"(?:^| )({re.escape(val)})(?: |$)", cand):
            matches.append((len(val.split()), len(val), letter))

    if not matches:
        return None

    # Prefer the longest option phrase, useful for cases like "black" vs "black and white".
    matches.sort(reverse=True)
    best = matches[0]
    tied = [m for m in matches if m[:2] == best[:2]]
    if len(tied) == 1:
        return best[2]
    return None


def semantic_sentence_match(text, options):
    # Fall back to the last few answer-like sentences. Negative option-enumeration
    # sentences are down-weighted rather than treated as positive answers.
    pieces = [
        p.strip()
        for p in re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
        if p.strip()
    ]
    if not pieces:
        return None

    scored = []
    tail = pieces[-6:]
    for idx, piece in enumerate(tail):
        pnorm = normalize(piece)
        negative = any(
            phrase in pnorm
            for phrase in [
                "other options",
                "do not match",
                "does not match",
                "not present",
                "are not present",
                "not the",
            ]
        )
        for letter, value in options.items():
            val = normalize(value)
            if not val:
                continue
            if re.search(rf"(?:^| )({re.escape(val)})(?: |$)", pnorm):
                score = idx + 1
                if negative:
                    score -= 20
                # Positive answer constructions.
                if re.search(rf"\b(is|are|appears|looks|positioned|located)\b.*(?:^| )({re.escape(val)})(?: |$)", pnorm):
                    score += 10
                if "answer" in pnorm:
                    score += 20
                scored.append((score, len(val.split()), len(val), letter, piece))

    positive = [x for x in scored if x[0] > 0]
    if not positive:
        return None
    positive.sort(reverse=True)
    best = positive[0]
    # If two different letters have exactly the same score/phrase-length, leave unresolved.
    competitors = [x for x in positive if x[:3] == best[:3] and x[3] != best[3]]
    if competitors:
        return None
    return best[3]


def parse_option(text, options):
    text = text or ""

    # 1) Last boxed answer has highest precedence.
    boxes = re.findall(r"\\boxed\s*\{([^{}]*)\}", text, flags=re.I | re.S)
    if boxes:
        candidate = boxes[-1].strip()
        letter = parse_letter_prefix(candidate)
        if letter in options:
            return letter, "boxed_letter", candidate
        letter = exact_semantic_match(candidate, options)
        if letter:
            return letter, "boxed_semantic", candidate
        return None, "boxed_unresolved", candidate

    # 2) Last explicit FINAL ANSWER / ANSWER marker.
    markers = list(re.finditer(r"(?:FINAL\s+ANSWER|ANSWER)\s*:\s*", text, flags=re.I))
    if markers:
        candidate = text[markers[-1].end():].strip()
        letter = parse_letter_prefix(candidate)
        if letter in options:
            return letter, "answer_marker_letter", candidate
        letter = exact_semantic_match(candidate, options)
        if letter:
            return letter, "answer_marker_semantic", candidate

    # 3) Strict explicit final-answer letter anywhere near the tail.
    tail = text[-600:]
    explicit = list(re.finditer(
        r"(?:final\s+answer\s*(?:is)?|answer\s*(?:is)?)\s*[:\-]?\s*([A-D])(?:\s*[\).,:;\-]|\s+|$)",
        tail,
        flags=re.I,
    ))
    if explicit:
        letter = explicit[-1].group(1).upper()
        if letter in options:
            return letter, "explicit_final_letter", explicit[-1].group(0)

    # 4) Semantic fallback over the last answer-like sentences.
    letter = semantic_sentence_match(text, options)
    if letter:
        return letter, "semantic_sentence", ""

    return None, "unresolved", ""


def transition(base_correct, off_correct):
    if base_correct and off_correct:
        return "correct->correct"
    if base_correct and not off_correct:
        return "correct->wrong"
    if not base_correct and off_correct:
        return "wrong->correct"
    return "wrong->wrong"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main():
    root = Path(__file__).resolve().parents[1]
    vlmeval_dir = root / "third_party" / "VLMEvalKit"
    if str(vlmeval_dir) not in sys.path:
        sys.path.insert(0, str(vlmeval_dir))

    from vlmeval.dataset import build_dataset

    src = root / "results" / "causal_ablation" / "vstar_latent_off_all_triggered.jsonl"
    out_jsonl = root / "results" / "causal_ablation" / "vstar_latent_off_all_triggered_rescored.jsonl"
    out_summary = root / "results" / "causal_ablation" / "vstar_latent_off_all_triggered_rescored_summary.json"

    if not src.is_file():
        raise FileNotFoundError(src)

    records = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = build_dataset("VStarBench")

    rescored = []
    changed_parse_count = 0

    for r in records:
        pos = int(r["dataset_position"])
        row = dataset.data.iloc[pos]
        options = option_values(row)
        gt = str(row["answer"]).strip().upper()

        base_pred, base_method, base_candidate = parse_option(r["baseline_raw_text"], options)
        off_pred, off_method, off_candidate = parse_option(r["latent_off_raw_text"], options)

        base_correct = base_pred == gt if base_pred is not None else False
        off_correct = off_pred == gt if off_pred is not None else False

        if (
            base_pred != r.get("baseline_predicted_option_diagnostic")
            or off_pred != r.get("latent_off_predicted_option_diagnostic")
        ):
            changed_parse_count += 1

        rr = dict(r)
        rr.update({
            "options": options,
            "rescored_ground_truth": gt,
            "rescored_baseline_pred": base_pred,
            "rescored_baseline_method": base_method,
            "rescored_baseline_candidate": base_candidate,
            "rescored_baseline_correct": base_correct,
            "rescored_latent_off_pred": off_pred,
            "rescored_latent_off_method": off_method,
            "rescored_latent_off_candidate": off_candidate,
            "rescored_latent_off_correct": off_correct,
            "rescored_transition": transition(base_correct, off_correct),
        })
        rescored.append(rr)

    out_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rescored),
        encoding="utf-8",
    )

    transitions = Counter(r["rescored_transition"] for r in rescored)
    base_correct_n = sum(r["rescored_baseline_correct"] for r in rescored)
    off_correct_n = sum(r["rescored_latent_off_correct"] for r in rescored)
    unresolved_base = [r for r in rescored if r["rescored_baseline_pred"] is None]
    unresolved_off = [r for r in rescored if r["rescored_latent_off_pred"] is None]

    summary = {
        "dataset": "VStarBench",
        "n": len(rescored),
        "source": str(src),
        "scorer": "deterministic option-aware local rescoring; not official API judge",
        "changed_parse_samples": changed_parse_count,
        "baseline_correct_count": base_correct_n,
        "baseline_accuracy": base_correct_n / len(rescored),
        "latent_off_correct_count": off_correct_n,
        "latent_off_accuracy": off_correct_n / len(rescored),
        "accuracy_delta_off_minus_baseline": (off_correct_n - base_correct_n) / len(rescored),
        "transitions": dict(transitions),
        "mcnemar_exact_two_sided_p": mcnemar_exact(
            transitions["correct->wrong"],
            transitions["wrong->correct"],
        ),
        "unresolved_baseline_count": len(unresolved_base),
        "unresolved_latent_off_count": len(unresolved_off),
        "unresolved_baseline_positions": [r["dataset_position"] for r in unresolved_base],
        "unresolved_latent_off_positions": [r["dataset_position"] for r in unresolved_off],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("========== RESCORED SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n========== CHANGED / UNRESOLVED CASES ==========")
    for r in rescored:
        old_base = r.get("baseline_predicted_option_diagnostic")
        old_off = r.get("latent_off_predicted_option_diagnostic")
        new_base = r["rescored_baseline_pred"]
        new_off = r["rescored_latent_off_pred"]
        if old_base != new_base or old_off != new_off or new_base is None or new_off is None:
            print(
                f"pos={r['dataset_position']:3d} cat={r['category']:<20s} gt={r['rescored_ground_truth']} "
                f"base {old_base}->{new_base} ({r['rescored_baseline_method']}) "
                f"off {old_off}->{new_off} ({r['rescored_latent_off_method']}) "
                f"transition={r['rescored_transition']}"
            )

    print("\nVSTAR_OPTION_AWARE_RESCORE_PASS=True")


if __name__ == "__main__":
    main()
