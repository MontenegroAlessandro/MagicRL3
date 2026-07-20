import random
from datasets import Dataset


SYSTEM = (
    "You are a math assistant. Reason step by step, "
    r"then box your final answer with: \boxed{<number>}"
)


def make_dataset(n_samples: int, seed: int) -> tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    prompts, solutions = [], []
    for _ in range(n_samples):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        prompts.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"What is {a} + {b}?"},
        ])
        solutions.append(str(a + b))
    ds = Dataset.from_dict({"prompt": prompts, "solution": solutions})
    split = ds.train_test_split(test_size=0.1, seed=seed)
    return split["train"], split["test"]
