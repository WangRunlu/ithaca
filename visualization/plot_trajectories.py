import os, numpy as np, matplotlib.pyplot as plt

def plot_teacher_student(gng_npz: str, teacher_npz: str, policy_path: str, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    G = np.load(gng_npz); nodes, edges = G["nodes"], G["edges"]
    T = np.load(teacher_npz); coords = T["coords"]
    plt.figure(figsize=(8,6))
    plt.plot(coords[:,0], coords[:,1], linewidth=2)
    plt.scatter(nodes[:,0], nodes[:,1], s=4)
    for (i,j) in edges:
        p0, p1 = nodes[i], nodes[j]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], alpha=0.2)
    plt.title("Teacher trajectory on GNG graph (示意)")
    out = os.path.join(out_dir, "teacher_trajectory.png")
    plt.savefig(out, dpi=160, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")
