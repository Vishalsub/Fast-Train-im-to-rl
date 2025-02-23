import gymnasium as gym
from pick_and_place import MujocoFetchPickAndPlaceEnv  # Import your environment

# Initialize the environment
env = MujocoFetchPickAndPlaceEnv(reward_type="dense")

# Test for 10 episodes
for episode in range(10):
    state = env.reset()
    done = False
    step_count = 0

    print(f"🚀 Starting Test Episode {episode+1}")

    while not done and step_count < 100:  # Run max 100 steps per episode
        action = env.action_space.sample()  # Take random actions
        next_state, reward, done, _, info = env.step(action)

        print(f"Step {step_count}: Reward={reward}, Done={done}")
        step_count += 1

env.close()
print("✅ Environment test completed successfully!")
