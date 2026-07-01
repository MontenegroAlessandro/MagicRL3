import os
os.environ["HF_HOME"] = "/storage/fis1/hf_cache"

import sys
import torch
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hydra.utils import get_method
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Componenti del nome del gruppo wandb: ogni entry è (etichetta, valore).
# Modifica/aggiungi/rimuovi voci qui per cambiare cosa compare nel group name.
def build_group_name(exp: DictConfig) -> str:
    rew_label = "+".join(exp.reward_fns)
    parts = [
        "GRPO",
        f"[{exp.task_name}]",
        f"rew={rew_label}",
        f"bs{exp.per_device_train_batch_size}",
        f"g{exp.num_generations}",
        f"lr{exp.learning_rate}",
        f"beta{exp.beta}",
        f"eps{exp.epsilon}",
    ]
    return " ".join(parts)


@hydra.main(version_base=None, config_path=".", config_name="conf_grpo")
def main(cfg: DictConfig):
    exp = cfg.experiment
    torch.set_num_threads(exp.num_threads)

    task = importlib.import_module(f"tasks.{exp.task_name}")
    reward_fns = [get_method(fn) for fn in exp.reward_fns]
    reward_weights = list(exp.reward_weights) if exp.reward_weights is not None else None
    train_dataset, eval_dataset = task.make_dataset(exp.n_samples, exp.seed)

    group_name = build_group_name(exp)
    run_name = f"{group_name} seed={exp.seed}"

    conf = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=conf,
        group=group_name,
        name=run_name,
        tags=list(cfg.wandb.tags),
        notes=cfg.wandb.notes or None,
        mode=cfg.wandb.mode,
    )

    grpo_cfg = GRPOConfig(
        output_dir=f"{exp.dir_name}/grpo/{run.id}",
        seed=exp.seed,
        num_train_epochs=exp.num_train_epochs,
        max_steps=exp.max_steps,
        per_device_train_batch_size=exp.per_device_train_batch_size,
        gradient_accumulation_steps=exp.gradient_accumulation_steps,
        learning_rate=exp.learning_rate,
        lr_scheduler_type=exp.lr_scheduler_type,
        warmup_ratio=exp.warmup_ratio,
        warmup_steps=exp.warmup_steps,
        optim=exp.optim,
        weight_decay=exp.weight_decay,
        max_grad_norm=exp.max_grad_norm,
        num_generations=exp.num_generations,
        num_iterations=exp.num_iterations,
        beta=exp.beta,
        epsilon=exp.epsilon,
        epsilon_high=exp.epsilon_high,
        delta=exp.delta,
        loss_type=exp.loss_type,
        scale_rewards=exp.scale_rewards,
        importance_sampling_level=exp.importance_sampling_level,
        mask_truncated_completions=exp.mask_truncated_completions,
        top_entropy_quantile=exp.top_entropy_quantile,
        max_completion_length=exp.max_completion_length,
        temperature=exp.temperature,
        top_p=exp.top_p,
        top_k=exp.top_k,
        min_p=exp.min_p,
        repetition_penalty=exp.repetition_penalty,
        generation_batch_size=exp.generation_batch_size,
        steps_per_generation=exp.steps_per_generation,
        shuffle_dataset=exp.shuffle_dataset,
        sync_ref_model=exp.sync_ref_model,
        ref_model_mixup_alpha=exp.ref_model_mixup_alpha,
        ref_model_sync_steps=exp.ref_model_sync_steps,
        eval_strategy=exp.eval_strategy,
        eval_steps=exp.eval_steps,
        per_device_eval_batch_size=exp.per_device_eval_batch_size,
        num_generations_eval=exp.num_generations_eval,
        logging_steps=exp.logging_steps,
        report_to="wandb",
        run_name=run.name,
        save_strategy="no",
        use_cpu=True,
        log_completions=True,
        num_completions_to_print=4,
    )

    trainer = GRPOTrainer(
        model=exp.model_name,
        reward_funcs=reward_fns,
        reward_weights=reward_weights,
        args=grpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
