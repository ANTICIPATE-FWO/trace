import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from policy_explainability.setting import initialize_setting
from policy_explainability.morl_baselines.multi_policy.ipro.ipro import IPRO
from policy_explainability.analysis import save_traj, visualize_pareto

def main():
    env_id, method = "deep-sea-treasure-v0", "ipro"
    env, eval_env, ref_point, file_prefix = initialize_setting(env_id=env_id)
    datapath =f"data/{file_prefix}_{method}.json"

    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=1e-9,
        max_iterations=50,
        iter_total_timesteps=500_000,
        learning_rate=2.5e-4,
        device="cpu",
        log=False,
        seed=0,
    )

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=ref_point,
        deterministic=False,
    )
    save_traj(pareto_set, path=datapath)

    pareto_front = ipro.get_pareto_front()
    print(f'\nPareto front points: {len(pareto_front)}')

    # only for deep sea treasure
    print(f'Ground truth: {np.sum(eval_env.unwrapped.sea_map > 0)} points')
    visualize_pareto(pareto_front)


if __name__ == "__main__":
    main()