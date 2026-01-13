import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

def make_env_factory(env_id:str):
    def _factory(): return mo_gym.make(env_id)
    return _factory


def initialize_setting(env_id:str = "deep-sea-treasure-v0"):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id)]))
    eval_env = mo_gym.make(env_id)
    from policy_explainability.setting.env_data import env_data
    return env, eval_env, env_data[env_id]["ref_point"], env_data[env_id]["file_prefix"]


def dst_ground_truth(env):
    raise NotImplementedError
