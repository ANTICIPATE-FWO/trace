import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

def make_env_factory(env_id: str):
    def _factory(): return mo_gym.make(env_id)
    return _factory


def initialize_setting(env_id: str):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id)]))
    eval_env = mo_gym.make(env_id)
    return env, eval_env