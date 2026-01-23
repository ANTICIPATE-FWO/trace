from collections import Counter
import numpy as np
np.set_printoptions(precision=2)


def cluster_report(actions: np.ndarray, rewards: np.ndarray, id: int, max_examples: int = 5):
    n_episodes = actions.shape[0]

    episode_lengths = np.sum(actions != -1, axis=1)
    valid_actions = actions[actions != -1]
    action_counts = Counter(valid_actions.tolist())
    reward_range = np.stack([rewards.min(axis=0), rewards.max(axis=0)],axis=1)
    reward_range = [[round(mn, 2), round(mx, 2)] for mn, mx in reward_range]
    example_indices = np.linspace(0, n_episodes - 1, min(max_examples, n_episodes), dtype=int)

    print("=" * 60)
    print(f"CLUSTER {id} BEHAVIOR REPORT")
    print("=" * 60)

    print(f"Number of episodes: {n_episodes}")
    print()

    print("Episode length:")
    print(f"  Mean   : {episode_lengths.mean():.2f}")
    print(f"  Std    : {episode_lengths.std():.2f}")
    print(f"  Min/Max: {episode_lengths.min()} / {episode_lengths.max()}")
    print()

    print("Action distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.most_common():
        frac = count / total_actions
        print(f"  Action {action}: {count} ({frac:.2%})")
    print()

    print("Rewards dimension: ", rewards.shape[1])
    print(f"  Mean   : {rewards.mean(axis=0)}")
    print(f"  Std    : {rewards.std(axis=0)}")
    print(f"  Range  : {reward_range}")
    print()

    print("Example action sequences:")
    for idx in example_indices:
        seq = actions[idx]
        seq = seq[seq != -1]
        print(f"  Episode {idx}: {seq.tolist()}")

    print("=" * 60)
