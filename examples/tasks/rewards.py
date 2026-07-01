
def format_reward(completions, **_) -> list[float]:
    """1.0 se la completion contiene \\boxed{}, 0.0 altrimenti."""
    raise NotImplementedError


def length_penalty(completions, **_) -> list[float]:
    """Penalizza completion troppo lunghe."""
    raise NotImplementedError
