
import re

from trl.rewards import accuracy_reward as trl_accuracy_reward


def accuracy_reward(completions, solution, weight=1.0, **kwargs) -> list[float | None]:
    """
    Wrapper pesato di trl.rewards.accuracy_reward (preserva i None).
    """
    rewards = trl_accuracy_reward(completions=completions, solution=solution, **kwargs)
    return [None if r is None else weight * r for r in rewards]


def format_reward(completions, weight=1.0, **_) -> list[float]:
    """
    weight se la completion contiene \\boxed{}, 0.0 altrimenti.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]

        # Check if the completion contains a \boxed{} structure
        if re.search(r"\\boxed\{[^}]+\}", text):
            rewards.append(weight)  # Partial reward for correct formatting
        else:
            rewards.append(0.0)

    return rewards


def length_penalty(completion_ids, weight=1.0, max_len=512, soft_len=64, **_) -> list[float]:
    """
    Penalizza le completion troppo lunghe (soft overlong punishment, DAPO eq. 13):
    0.0 entro il margine di sicurezza, penalità lineare avvicinandosi a max_len,
    -weight al raggiungimento o superamento di max_len.
    """
    threshold = max_len - soft_len
    rewards = []
    for ids in completion_ids:
        length = len(ids)
        if length <= threshold:
            rewards.append(0.0)
        elif length <= max_len:
            rewards.append(weight * (threshold - length) / soft_len)
        else:
            rewards.append(-weight)
    return rewards
