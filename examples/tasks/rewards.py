
import re

def format_reward(completions, **_) -> list[float]:
    """
    1.0 se la completion contiene \\boxed{}, 0.0 altrimenti.
    todo: pesare la rewward, non può essere maggiore della reward per la correttezza della risposta.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion[0], dict) else completion[0]

        # Check if the completion contains a \boxed{} structure
        if re.search(r"\\boxed\{[^}]+\}", text):
            rewards.append(1.0)  # Partial reward for correct formatting
        else:
            rewards.append(0.0)


def length_penalty(completions, **_) -> list[float]:
    """Penalizza completion troppo lunghe."""
    raise NotImplementedError
