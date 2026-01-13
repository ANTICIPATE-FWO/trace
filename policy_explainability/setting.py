import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

env_ids = {
    "deep sea treasure": "deep-sea-treasure-v0",
    "minecart": "minecart-v0",
}

def make_env_factory(env_id:str):
    def _factory(): return mo_gym.make(env_id)
    return _factory


def initialize_setting(env_id:str = "deep-sea-treasure-v0"):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id)]))
    eval_env = mo_gym.make(env_id)

    #file path
    #ref point
    return env, eval_env


def assign_ref_point(env_id:str):
    pass

def dst_ground_truth(env):
    raise NotImplementedError
