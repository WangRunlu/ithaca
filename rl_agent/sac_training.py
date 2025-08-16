import os, json, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from .env_graph_nav import GraphNavEnv

def _make_env(nodes, edges, edge_attr):
    def _thunk(): return GraphNavEnv(nodes, edges, edge_attr)
    return _thunk

def train_sac_agent(gng_npz: str, edge_json: str, teacher_npz: str, agent_dir: str, total_timesteps=20000):
    g = np.load(gng_npz); nodes, edges = g["nodes"], g["edges"]
    with open(edge_json, "r", encoding="utf-8") as f:
        edge_attr = json.load(f)["edge_attr"]
    env = DummyVecEnv([_make_env(nodes, edges, edge_attr)])
    model = SAC("MlpPolicy", env, verbose=0, tensorboard_log=os.path.join(agent_dir, "tb"))
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    out_path = os.path.join(agent_dir, "sac_policy.zip"); model.save(out_path)
    print(f"[ok] SAC policy -> {out_path}")
    return out_path
