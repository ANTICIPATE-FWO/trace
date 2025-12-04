from typing import Optional
import mo_gymnasium as mo_gym

class EnvWrapper:
    def __init__(self, name:str="deep-sea-treasure-v0", render:str="human"):
        self.name, self.render = name, render
        self.env = mo_gym.make(self.name, render_mode=render)

    def action(self, policy:Optional[str]=None):
        return self.env.action_space.sample()

    def episode(self, max_steps:int=100):
        obs, _ = self.env.reset()
        print("Initial observation:", obs)

        terminated, truncated, step = False, False, 0

        while not (terminated or truncated) and step < max_steps:
            action = self.action()
            obs, reward, terminated, truncated, _ = self.env.step(action)

            print(f"Step: {step} Action: {action}, Reward: {reward}, Obs: {obs.round(2)}")
            step += 1

    def close(self):
        self.env.close()

def main():
    env_names = ["deep-sea-treasure-v0", "minecart-v0", "mo-reacher-v5"]
    env = EnvWrapper(env_names[2])
    env.episode()
    env.close()

    return

if __name__ == "__main__": main()