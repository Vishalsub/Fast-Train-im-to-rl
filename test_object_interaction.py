import gymnasium as gym
import numpy as np
import time
from pick_and_place import MujocoFetchPickAndPlaceEnv  # Import your environment

# Initialize the environment
env = MujocoFetchPickAndPlaceEnv(reward_type="dense", render_mode="human")


# Reset the environment
state = env.reset()
done = False

print("🚀 Starting Object Interaction Test")

# Test pushing the object
print("🔹 Testing: Pushing the Object")
for step in range(20):
    action = np.array([0.05, 0, 0, 0])  # Move in +X direction
    next_state, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.05)  # Add delay to visualize the movement
print("✅ Pushing test completed.")

# Test reaching phase
print("🔹 Testing: Reaching and Grasping")
for step in range(20):
    action = np.array([0, 0, 0.1, 0])  # Move the gripper upward
    next_state, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.05)
print("✅ Reaching test completed.")

# Test closing gripper
print("🔹 Testing: Closing the Gripper")
for step in range(10):
    action = np.array([0, 0, 0, -0.2])  # Close the gripper
    next_state, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.05)
print("✅ Grasping test completed.")

# Test lifting the object
print("🔹 Testing: Lifting the Object")
for step in range(20):
    action = np.array([0, 0, 0.05, 0])  # Lift the object
    next_state, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.05)
print("✅ Lifting test completed.")

env.close()
print("🎯 Object Interaction Test Completed Successfully!")
