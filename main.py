import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from env_abstraction import initialize_envs
from morl_baselines.multi_policy.ipro.ipro import IPRO
from tracking.track import save_traj
import numpy as np

def main():
    env, eval_env = initialize_envs()
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
    save_traj(pareto_set)

    pareto_front = ipro.get_pareto_front()
    print(f'Pareto front: {len(pareto_front)} points')
    print(f'Ground truth: {np.sum(eval_env.unwrapped.sea_map > 0)} points')


if __name__ == "__main__":
    main()