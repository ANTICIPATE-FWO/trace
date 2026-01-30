import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

from trace.core.metadata import env_metadata


def make_env_factory(env_id:str):
    def _factory(): return mo_gym.make(env_id)
    return _factory


def initialize_setting(env_id:str = "deep-sea-treasure-v0"):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id)]))
    eval_env = mo_gym.make(env_id)
    return (
        env,
        eval_env,
        env_metadata[env_id]["ref_point"],
        env_metadata[env_id]["file_prefix"],
        env_metadata[env_id]["actions"]
    )
