import re
from datasets import load_dataset, Dataset


def make_dataset(n_samples: int, seed: int) -> tuple[Dataset, Dataset]:
    ds = load_dataset("openai/gsm8k", "main")

    train = ds["train"].shuffle(seed=seed).select(range(min(n_samples, len(ds["train"]))))
    test  = ds["test"].select(range(min(200, len(ds["test"]))))

    def format_sample(example):
        nums = re.findall(r"####\s*(-?\d+)", example["answer"])
        question = f"{example['question']}\n\nEnd your response with the final answer in the format: #### <number>"
        return {
            "prompt": [{"role": "user", "content": question}],
            "answer": int(nums[0]) if nums else None,
        }

    train = train.map(format_sample).filter(lambda x: x["answer"] is not None)
    test  = test.map(format_sample).filter(lambda x: x["answer"] is not None)

    return train, test


def _extract_answer(text: str) -> int | None:
    # privilegia il delimitatore richiesto nel prompt, con fallback sull'ultimo
    # numero nel testo per non azzerare il reward finché il modello non impara il formato
    match = re.findall(r"####\s*(-?\d+)", text)
    if match:
        return int(match[-1])
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def reward_fn(completions, prompts, answer, **kwargs) -> list[float]:
    """Reward continua su errore relativo: scale-invariante su GSM8K."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]
        predicted = _extract_answer(text)
        if predicted is None:
            rewards.append(0.0)
        elif predicted == expected:
            rewards.append(1.0)
        else:
            rel_error = abs(predicted - expected) / max(abs(expected), 1)
            rewards.append(max(0.0, 1.0 - rel_error))
    return rewards
