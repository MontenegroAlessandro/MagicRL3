from datasets import load_dataset, Dataset


SYSTEM = (
    "You are a math assistant. Write only the equation that solves the problem, "
    r"then the final answer as: \boxed{<number>}. No words, no explanations."
    "\nExample: (5 + 3) * 2 = 16, \boxed{16}"
)


def make_dataset(n_samples: int, seed: int) -> tuple[Dataset, Dataset]:
    ds = load_dataset("ChilleD/MultiArith")

    train = ds["train"].shuffle(seed=seed).select(range(min(n_samples, len(ds["train"]))))
    test  = ds["test"].select(range(min(200, len(ds["test"]))))

    def format_sample(example):
        # le question del dataset hanno spazi spuri ai bordi, final_ans è già il numero come stringa
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": example["question"].strip()},
            ],
            "solution": example["final_ans"].strip(),
        }

    train = train.map(format_sample).remove_columns(["question", "final_ans"])
    test  = test.map(format_sample).remove_columns(["question", "final_ans"])

    return train, test
