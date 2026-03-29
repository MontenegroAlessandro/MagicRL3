import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ==== CONFIG ====
ENTITY = "rventurelli-politecnico-di-milano"
PROJECT = "sb3-a2c-half-cheetah"

METRIC = "rollout/ep_rew_mean"
X_AXIS = "global_step"
SMOOTH = 0.0

REQUIRED_TAGS = ["rt_validation"]   # filtra esperimenti

SELECTED_GROUPS = [
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=1 opc=False",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=2 opc=False",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=5 opc=False",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=10 opc=False",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=2 opc=True",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=5 opc=True",
    "RT-A2C 8x128 lr=0.0014 ent=0.002 gae_lambda=0.95 w=10 opc=True",
]

# =================

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# =================
# DEBUG: TAG → GROUPS
# =================
print("\n===== TAG → GROUPS =====")
tag_groups = defaultdict(set)

for run in runs:
    if run.group is None:
        continue
    for tag in run.tags:
        tag_groups[tag].add(run.group)

for tag in sorted(tag_groups):
    print(f"\n{tag}")
    for g in sorted(tag_groups[tag]):
        print(f"  - {g}")

# =================
# FILTER + GROUPING (per learning rate)
# =================
groups = defaultdict(list)

for run in runs:
    tags = run.tags
    group = run.group

    # filtro group
    if len(SELECTED_GROUPS) > 0 and group not in SELECTED_GROUPS:
        continue

    # filtro tag
    if not all(tag in tags for tag in REQUIRED_TAGS):
        continue

    key = run.group if run.group is not None else "NO_GROUP"

    groups[key].append(run)

print("\n===== GRUPPI DOPO FILTRO =====")
for g, rs in groups.items():
    print(f"{g}: {len(rs)} runs")

# =================
# SMOOTH
# =================
def smooth(y, weight):
    if weight == 0:
        return y
    y_smooth = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        y_smooth.append(last)
    return np.array(y_smooth)

# =================
# PLOT
# =================
plt.figure(figsize=(10, 6))

for group_name, group_runs in groups.items():

    all_curves = []
    min_len = np.inf
    x_ref = None

    for run in group_runs:
        history = run.history(keys=[X_AXIS, METRIC], pandas=True)

        if len(history) == 0:
            continue

        x = history[X_AXIS].values
        y = history[METRIC].values

        # rimuovi NaN
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(y) == 0:
            continue

        y = smooth(y, SMOOTH)

        if x_ref is None:
            x_ref = x

        min_len = min(min_len, len(y))
        all_curves.append(y)

    if len(all_curves) < 2:
        print(f"Skip {group_name}: <2 runs validi")
        continue

    # allineamento
    all_curves = [c[:min_len] for c in all_curves]
    all_curves = np.array(all_curves)

    x = x_ref[:min_len]

    mean = np.mean(all_curves, axis=0)
    std = np.std(all_curves, axis=0)
    ci = 1.96 * std / np.sqrt(len(all_curves)) 

    plt.plot(x, mean, label=group_name)
    plt.fill_between(x, mean - ci, mean + ci, alpha=0.2)

plt.xlabel(X_AXIS)
plt.ylabel(METRIC)
plt.title("Mean ± 95% CI over seeds")
plt.legend()
plt.grid()

plt.savefig("plot.png", dpi=150, bbox_inches="tight")
print("Plot salvato in plot.png")