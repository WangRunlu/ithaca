import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GraphNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, nodes, edges, edge_attr):
        super().__init__()
        self.nodes, self.edges, self.edge_attr = nodes, edges, edge_attr
        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)  # ax, ay
        self.dt = 0.2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = int(np.argmin(self.nodes[:,0]+self.nodes[:,1]))
        t = int(np.argmax(self.nodes[:,0]+self.nodes[:,1]))
        self.goal = self.nodes[t]
        self.state = np.array([self.nodes[s,0], self.nodes[s,1], 0.0, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        ax, ay = action.astype(np.float32)
        x, y, vx, vy = self.state
        vmax = 12.0
        vx = np.clip(vx + ax*self.dt, -vmax, vmax)
        vy = np.clip(vy + ay*self.dt, -vmax, vmax)
        x = x + vx*self.dt; y = y + vy*self.dt
        self.state = np.array([x,y,vx,vy], dtype=np.float32)
        dist = np.linalg.norm(self.goal - self.state[:2])
        r = -dist*0.05 - 0.01*(vx*vx+vy*vy)
        terminated = dist < 3.0
        truncated = False
        return self.state, r, terminated, truncated, {}
