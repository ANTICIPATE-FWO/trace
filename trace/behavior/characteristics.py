from collections import Counter
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
np.set_printoptions(precision=2)

def behavior_report(actions: np.ndarray, rewards: np.ndarray, ind: int, max_examples: int = 5):
    n_episodes = actions.shape[0]

    episode_lengths = np.sum(actions != -1, axis=1)
    valid_actions = actions[actions != -1]
    action_counts = Counter(valid_actions.tolist())

    example_indices = np.linspace(0, n_episodes - 1, min(max_examples, n_episodes), dtype=int)

    print("=" * 60)
    print(f"BEHAVIOR CLUSTER {ind} REPORT")
    print("=" * 60)

    print(f"Number of episodes: {n_episodes}")
    print()

    print("Episode length:")
    print(f"  Mean ± Std: {episode_lengths.mean()} ± {episode_lengths.std()}")
    print(f"  Range     : {episode_lengths.min()} / {episode_lengths.max()}")
    print()

    print("Action distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.most_common():
        frac = count / total_actions
        print(f"  Action {action}: {count} ({frac:.2%})")
    print()

    print("Example action sequences:")
    for idx in example_indices:
        seq = actions[idx]
        seq = seq[seq != -1]
        print(f"  Episode {idx}: {seq.tolist()}")

    print("=" * 60)


def reward_report(actions:np.ndarray, rewards: np.ndarray, ind: int):
    minmax = np.stack([rewards.min(axis=0), rewards.max(axis=0)], axis=1)
    minmax = [[round(mn, 2), round(mx, 2)] for mn, mx in minmax]
    formatted = "\t".join(f"{m:.2f} ± {s:.2f}" for m, s in zip(rewards.mean(axis=0), rewards.std(axis=0)))

    print("=" * 60)
    print(f"REWARD CLUSTER {ind} REPORT")
    print("=" * 60)
    print("Rewards dimension: ", rewards.shape[1])
    print(f"  Mean ± Std: [{formatted}]")
    print(f"  Range     : {minmax}")
    print()


    print("=" * 60)

def tree_rules(obs, acs, feature_names: list = None):
    if feature_names is None: feature_names = ['x', 'y']
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2)
    tree.fit(obs, acs)
    return export_text(tree, feature_names=feature_names)