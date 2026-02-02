import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.chdir('..')

from trace.core import initialize_setting, dst_ground_truth
from trace.morl_baselines.multi_policy.ipro.ipro import IPRO
from trace.core import TrajectoryManager

def main():
    env_id, method = "deep-sea-treasure-v0", "ground_truth"
    env, eval_env, ref_point, file_prefix, actions = initialize_setting(env_id=env_id)
    filepath =f"data/{file_prefix}_{method}.json"

    TrajectoryManager(env_id).load([dst_ground_truth(eval_env.unwrapped.sea_map)]).save(filepath)
    print(f'Saved ground truth trajectories in {filepath}.')

    method = "ipro"
    filepath = f"data/{file_prefix}_{method}.json"
    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=1e-6,
        max_iterations=50,
        iter_total_timesteps=500_000,
        num_steps=256,
        learning_rate=2.5e-4,
        device="cpu",
        log=False,
        seed=0,
    )
    print('Initialized environment and algorithm.')

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=ref_point,
        deterministic=False,
    )
    TrajectoryManager(env_id).load([traj for _, _, traj in pareto_set]).save(filepath)
    print(f'Saved Pareto trajectories in {filepath}.')
    print(f'\nIPRO Pareto front points: {ipro.get_pareto_front()}')



if __name__ == "__main__":
    main()