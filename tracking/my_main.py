import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
import numpy as np

from morl_baselines.multi_policy.ipro.ipro import IPRO
from my_utils import make_env, visualize_pareto, visualize_dst_map


def main():
    env = MOSyncVectorEnv([make_env])
    eval_env = mo_gym.make("deep-sea-treasure-v0")
    ref_point = np.array([-100.0, -100.0])


    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=1e-9,
        iter_total_timesteps=50_000,
        learning_rate=2.5e-4,
        device="cpu",
        log=False,
        seed=0,
    )

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=ref_point,
        deterministic=True,
    )

    pareto_front = ipro.get_pareto_front()

    print("\nPareto front points:")
    for p in pareto_front: print(p)

    print(f'Ground truth: {np.sum(eval_env.unwrapped.sea_map > 0)} points')

    from json import dumps
    all_trajectories = [traj for _, _, traj in pareto_set]
    #implement filtering function after forking

    #with open("/Users/emilypalaska/Documents/policy_explainability/data/data.json", "w") as f:
    #    f.write(dumps(all_trajectories, indent=2))

    #visualize_pareto(pareto_front)


if __name__ == "__main__":
    main()
