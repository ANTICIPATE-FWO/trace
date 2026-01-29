import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

env_data = {
    "deep-sea-treasure-v0": {
        "ref_point": [-100.0, -100.0],
        "file_prefix": "dst",
        "actions": [0,1,2,3]
    },
    "minecart-v0": {
        "ref_point": [-100.0, -100.0, -100.0],
        "file_prefix": "mc"
    }
}


def make_env_factory(env_id:str):
    def _factory(): return mo_gym.make(env_id)
    return _factory


def initialize_setting(env_id:str = "deep-sea-treasure-v0"):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id)]))
    eval_env = mo_gym.make(env_id)
    return env, eval_env, env_data[env_id]["ref_point"], env_data[env_id]["file_prefix"], env_data[env_id]["actions"]
