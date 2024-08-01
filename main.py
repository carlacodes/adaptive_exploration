import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco_viewer
# Create the environment
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env


MODEL_XML = """
<mujoco model="2D_navigation">
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        <body name="agent" pos="0 0 0.1">
            <geom name="agent_geom" type="sphere" size="0.1" rgba="0.0 0.6 1.0 1"/>
            <joint name="agent_x" type="slide" axis="1 0 0"/>
            <joint name="agent_y" type="slide" axis="0 1 0"/>
        </body>
        <body name="goal" pos="3 3 0.1">
            <geom name="goal_geom" type="sphere" size="0.1" rgba="1.0 0.0 0.0 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="agent_x" ctrlrange="-1 1"/>
        <motor joint="agent_y" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

class Navigation2DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(Navigation2DEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal')
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_shape = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.np_random = np.random.default_rng()
        self.render_mode = render_mode

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.data.qpos[:] = np.zeros_like(self.data.qpos)
        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        agent_pos = self.data.qpos[:2]
        goal_pos = self.data.xpos[self.goal_body_id][:2]
        return np.concatenate([agent_pos, goal_pos])

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = -np.linalg.norm(obs[:2] - obs[2:])
        done = reward > -0.1
        return obs, reward, done, {}, {}

    def render(self, mode='human'):
        if self.render_mode is None:
            raise ValueError("No render_mode was passed to the environment constructor.")
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



    # Test the Navigation2DEnv environment
if __name__ == "__main__":
    # Create the environment with render_mode
    env = make_vec_env(Navigation2DEnv, n_envs=1, env_kwargs={"render_mode": "human"})

    # Instantiate the TRPO agent
    model = TRPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("trpo_navigation2d")

    # Load the model
    model = TRPO.load("trpo_navigation2d")

    # Test the trained agent
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()

