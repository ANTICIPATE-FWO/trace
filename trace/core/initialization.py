import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from trace.core.minetrain import MinecartTrailWrapper

def make_env_factory(env_id: str, minetrain:bool=False):
    def _factory():
        env = mo_gym.make(env_id)
        if minetrain: return MinecartTrailWrapper(env)
        else: return env
    return _factory

def initialize_setting(env_id: str, minetrain:bool=False):
    env = MOSyncVectorEnv(iter([make_env_factory(env_id, minetrain=minetrain)]))
    eval_env = MinecartTrailWrapper(mo_gym.make(env_id)) if minetrain else mo_gym.make(env_id)
    return env, eval_env