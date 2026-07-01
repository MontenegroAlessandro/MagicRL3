import re
from datasets import Dataset


def make_dataset(n_samples: int, seed: int) -> tuple[Dataset, Dataset]:
    import random
    random.seed(seed)
    prompts, answers = [], []
    for _ in range(n_samples):
        a, b = random.randint(1, 50), random.randint(1, 50)
        prompts.append({"role": "user", "content": f"What is {a} + {b}? Reply with just the number."})
        answers.append(a + b)
    ds = Dataset.from_dict({"prompt": [[p] for p in prompts], "answer": answers})
    split = ds.train_test_split(test_size=0.1, seed=seed)
    return split["train"], split["test"]


def reward_fn(completions, prompts, answer, **kwargs) -> list[float]:
    """Reward 1.0 se conciso e corretto, 0.2 se corretto ma verboso, 0.0 se sbagliato."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]
        nums = re.findall(r"-?\d+", text)
        predicted = int(nums[-1]) if nums else None
        correct = predicted == expected
        concise = text.strip() == str(expected)
        rewards.append(1.0 if concise else 0.2 if correct else 0.0)
    return rewards


def reward_fn_continuous(completions, prompts, answer, **kwargs) -> list[float]:
    """Reward continua: 1.0 se esatto, decade con l'errore."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]
        nums = re.findall(r"-?\d+", text)
        predicted = int(nums[-1]) if nums else None
        if predicted is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 / (1.0 + abs(predicted - expected)))
    return rewards
