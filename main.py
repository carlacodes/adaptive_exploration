import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer
import mujoco_viewer

# Define the XML model content
MODEL_XML = """
<mujoco model="2D_navigation">
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <!-- The floor plane -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>

        <!-- The agent (a point mass) -->
        <body name="agent" pos="0 0 0.1">
            <geom name="agent_geom" type="sphere" size="0.1" rgba="0.0 0.6 1.0 1"/>
            <joint name="agent_x" type="slide" axis="1 0 0"/>
            <joint name="agent_y" type="slide" axis="0 1 0"/>
        </body>

        <!-- The goal -->
        <body name="goal" pos="3 3 0.1">
            <geom name="goal_geom" type="sphere" size="0.1" rgba="1.0 0.0 0.0 1"/>
        </body>
    </worldbody>
    <actuator>
        <!-- Actuators to control the agent's movement -->
        <motor joint="agent_x" ctrlrange="-1 1"/>
        <motor joint="agent_y" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""



class Navigation2DEnv(gym.Env):
    def __init__(self):
        super(Navigation2DEnv, self).__init__()

        # Load the model and create the simulation
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Get the body index for the goal
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal')

        # Action space: 2D control (x, y)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: agent position (x, y) and goal position (x, y)
        obs_shape = 4  # 2 for agent position + 2 for goal position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        # Manually set the initial positions and velocities
        self.data.qpos[:] = np.zeros_like(self.data.qpos)  # or specific initial values
        self.data.qvel[:] = np.zeros_like(self.data.qvel)  # or specific initial values
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        # Get agent and goal positions
        agent_pos = self.data.qpos[:2]  # Assuming qpos contains the agent position
        goal_pos = self.data.xpos[self.goal_body_id][:2]  # Use xpos for body position
        return np.concatenate([agent_pos, goal_pos])

    def step(self, action):
        # Apply action to the simulation
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Get observation, reward, and done flag
        obs = self._get_obs()
        reward = -np.linalg.norm(obs[:2] - obs[2:])
        done = reward > -0.1  # Consider done if within 0.1 units of the goal
        return obs, reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            # Create a new viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        # Render the simulation
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Test the Navigation2DEnv environment
if __name__ == "__main__":
    # Create the environment
    env = Navigation2DEnv()

    # Reset the environment
    obs = env.reset()
    print(f"Initial Observation: {obs}")

    # Run a simulation loop
    for _ in range(1000):  # Adjust the number of steps as needed
        # Sample a random action
        action = env.action_space.sample()
        print(f"Action: {action}")

        # Take a step in the environment
        obs, reward, done, _ = env.step(action)
        print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

        # Render the environment
        env.render()

        if done:
            print("Reached the goal or done condition met!")
            break

    # Close the environment
    env.close()
