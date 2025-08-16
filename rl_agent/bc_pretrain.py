import numpy as np, os

def bc_pretrain_from_teacher(teacher_npz: str, agent_dir: str):
    data = np.load(teacher_npz); coords = data["coords"]
    np.save(os.path.join(agent_dir, "teacher_coords.npy"), coords)
    print("[ok] BC 预热数据已生成（简版）。")
