import re
import math
from datasets import load_dataset, Dataset


SYSTEM = (
    "You are a math assistant. Reason step by step, "
    r"then box your final answer with: \boxed{<number>}"
)


def make_dataset(n_samples: int, seed: int) -> tuple[Dataset, Dataset]:
    ds = load_dataset("openai/gsm8k", "main")

    train = ds["train"].shuffle(seed=seed).select(range(min(n_samples, len(ds["train"]))))
    test  = ds["test"].select(range(min(200, len(ds["test"]))))

    def format_sample(example):
        match = re.search(r"####\s*(-?[\d,]+)", example["answer"])
        solution = match.group(1).replace(",", "") if match else None
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": example["question"]},
            ],
            "solution": solution,
        }

    train = (
        train.map(format_sample)
        .filter(lambda x: x["solution"] is not None)
        .remove_columns(["question", "answer"])
    )
    test = (
        test.map(format_sample)
        .filter(lambda x: x["solution"] is not None)
        .remove_columns(["question", "answer"])
    )

    return train, test


def _extract_boxed(text: str) -> float | None:
    """Estrae il numero da \\boxed{} nella completion; fallback sull'ultimo numero nel testo."""
    m = re.search(r"\\boxed\{(-?[\d,\.]+)\}", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


def smooth_reward(completions, solution, log_extra=None, **_) -> list[float]:
    """
    Reward densa alternativa ad accuracy_reward.

    Decadimento esponenziale sull'errore relativo:
      - 1.0        risposta esatta (bonus +0.5 rispetto al max non-esatto)
      - 0.5 * math.exp(-2 * rel_error)  altrimenti  → max 0.5, ~0.40 a 11%, ~0.23 a 39%
      - 0.0        nessun numero estratto

    Usa come reward_fn nel config: reward_fn: smooth_reward
    """
    rewards = []
    answer_parsed = []
    gold_parsed = []
    for comp, sol in zip(completions, solution):
        text = comp[0]["content"] if isinstance(comp[0], dict) else comp[0]
        predicted = _extract_boxed(text)
        try:
            expected = float(sol)
        except (ValueError, AttributeError):
            rewards.append(0.0)
            answer_parsed.append(None)
            gold_parsed.append(None)
            continue
        answer_parsed.append(predicted)
        gold_parsed.append(expected)
        if predicted is None:
            rewards.append(0.0)
        elif predicted == expected:
            rewards.append(1.0)
        else:
            rel_error = abs(predicted - expected) / max(abs(expected), 1.0)
            rewards.append(0.5 * float(math.exp(-2.0 * rel_error)))
    if log_extra is not None:
        log_extra("reward", rewards)
        log_extra("answer_parsed", answer_parsed)
        log_extra("gold_parsed", gold_parsed)
    return rewards
