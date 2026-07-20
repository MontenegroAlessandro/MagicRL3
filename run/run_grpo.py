import os
os.environ["HF_HOME"] = "/storage/fis1/hf_cache"

import sys
import torch
import functools
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hydra.utils import get_method
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.rt_grpo import RT_GRPOTrainer


# algo_label è strutturale (deriva meccanicamente da window_length/is_weight_type, sempre
# obbligatori): calcolato qui. La parte variabile del group name (quali parametri distinguono le
# run di QUESTA campagna, come abbreviarli) è invece editoriale, non meccanica — vedi
# experiment.group_name_suffix, impostato nel launcher di ogni lancio (CLAUDE.md).
IS_WEIGHT_LABEL = {"naive": "N", "bh": "BH"}


def build_group_name(exp: DictConfig) -> str:
    if exp.window_length < 1:
        algo_label = "GRPO"
    else:
        algo_label = f"RT-GRPO-{IS_WEIGHT_LABEL[exp.is_weight_type]} w{exp.window_length}"
    return f"{algo_label} {exp.group_name_suffix}".strip()


def load_reward_fn(spec):
    fn = get_method(spec.path)
    kwargs = {k: v for k, v in spec.items() if k != "path"}
    if not kwargs:
        return fn
    partial_fn = functools.partial(fn, **kwargs)
    functools.update_wrapper(partial_fn, fn)  # TRL legge __name__ per il logging
    return partial_fn


def build_peft_config(exp: DictConfig) -> LoraConfig | None:
    if not exp.lora.enabled:
        return None
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=exp.lora.r,
        lora_alpha=exp.lora.alpha,
        lora_dropout=exp.lora.dropout,
        target_modules=exp.lora.target_modules,
        bias=exp.lora.bias,
    )


@hydra.main(version_base=None, config_path="configs", config_name="conf_grpo")
def main(cfg: DictConfig):
    exp = cfg.experiment
    torch.set_num_threads(exp.num_threads)

    task = importlib.import_module(f"llm_tasks.{exp.task_name}")
    reward_fns = [load_reward_fn(spec) for spec in exp.reward_fns]
    train_dataset, eval_dataset = task.make_dataset(exp.n_samples, exp.seed)
    if exp.eval_n_samples is not None:
        eval_dataset = eval_dataset.select(range(min(exp.eval_n_samples, len(eval_dataset))))


    group_name = build_group_name(exp)
    run_name = f"{group_name} seed={exp.seed}"

    conf = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=conf,
        group=group_name,
        name=run_name,
        tags=[*cfg.wandb.tags, exp.task_name],
        notes=cfg.wandb.notes or None,
        mode=cfg.wandb.mode,
        dir=exp.dir_name,
    )
    run_id = run.id

    grpo_cfg = GRPOConfig(
        output_dir=f"{exp.dir_name}/grpo/{run_id}",
        seed=exp.seed,
        model_init_kwargs={"dtype": exp.dtype},
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
        run_name=run_name,
        save_strategy="no",
        use_cpu=True,
        log_completions=True,
        num_completions_to_print=4,
    )

    peft_config = build_peft_config(exp)

    if exp.window_length < 1:
        trainer = GRPOTrainer(
            model=exp.model_name,
            reward_funcs=reward_fns,
            args=grpo_cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
    else:
        trainer = RT_GRPOTrainer(
            window_length=exp.window_length,
            is_weight_type=exp.is_weight_type,
            model=exp.model_name,
            reward_funcs=reward_fns,
            args=grpo_cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
