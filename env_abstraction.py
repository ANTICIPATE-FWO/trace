import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

def env_factory(env_id:str = "deep-sea-treasure-v0"):
    return mo_gym.make(env_id)

def initialize_envs(env_id:str = "deep-sea-treasure-v0"):
    env = MOSyncVectorEnv(iter([env_factory]))
    eval_env = mo_gym.make(env_id)
    return env, eval_env