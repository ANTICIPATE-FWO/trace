import mo_gymnasium as mo_gym

# Create the environment
env = mo_gym.make("deep-sea-treasure-v0", render_mode="human")

obs, info = env.reset()
done = False
step = 0

print("Initial observation:", obs)

while not done and step < 200:
    # Sample a random action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {step} | Action: {action}")
    print("Observation:", obs)
    print("Reward:", reward)  # <-- multi-objective reward
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    step += 1

env.close()
