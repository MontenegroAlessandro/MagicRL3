import re
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
import wandb

from trl import GRPOTrainer, GRPOConfig


def make_math_dataset(n_samples: int, seed: int) -> Dataset:
    import random
    random.seed(seed)
    prompts, answers = [], []
    for _ in range(n_samples):
        a, b = random.randint(1, 50), random.randint(1, 50)
        prompts.append({"role": "user", "content": f"What is {a} + {b}? Reply with just the number."})
        answers.append(a + b)
    return Dataset.from_dict({"prompt": [[p] for p in prompts], "answer": answers})


def reward_fn(completions, prompts, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]
        nums = re.findall(r"-?\d+", text)
        predicted = int(nums[-1]) if nums else None
        rewards.append(1.0 if predicted == expected else 0.0)
    return rewards


@hydra.main(version_base=None, config_path=".", config_name="conf_grpo")
def main(cfg: DictConfig):
    exp = cfg.experiment

    dataset = make_math_dataset(exp.n_samples, exp.seed)
    split = dataset.train_test_split(test_size=exp.eval_ratio, seed=exp.seed)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    conf = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project,
        config=conf,
        name=f"GRPO {exp.model_name.split('/')[-1]} seed={exp.seed}",
        tags=cfg.wandb.tags,
    )

    grpo_cfg = GRPOConfig(
        output_dir=f"{exp.dir_name}/grpo/{run.id}",
        num_train_epochs=exp.num_train_epochs,
        per_device_train_batch_size=exp.per_device_train_batch_size,
        num_generations=exp.num_generations,
        max_completion_length=exp.max_completion_length,
        learning_rate=exp.learning_rate,
        beta=exp.beta,
        epsilon=exp.epsilon,
        seed=exp.seed,
        report_to="wandb",
        run_name=run.name,
        logging_steps=exp.logging_steps,
        save_strategy="no",
        use_cpu=True,
    )

    trainer = GRPOTrainer(
        model=exp.model_name,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
