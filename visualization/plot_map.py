import os, json, numpy as np
import matplotlib.pyplot as plt

def plot_bev_maps(pom_npz: str, ph_json: str, gng_npz: str, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    P = np.load(pom_npz); xs, ys, prob = P["xs"], P["ys"], P["prob"]
    with open(ph_json, "r", encoding="utf-8") as f:
        ph = json.load(f)  # 未直接用，仅占位
    G = np.load(gng_npz); nodes, edges = G["nodes"], G["edges"]
    plt.figure(figsize=(8,6))
    plt.imshow(prob, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower')
    for (i,j) in edges:
        p0, p1 = nodes[i], nodes[j]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]])
    plt.scatter(nodes[:,0], nodes[:,1], s=5)
    plt.title("POM (prob) + GNG graph")
    out = os.path.join(out_dir, "bev_pom_gng.png")
    plt.savefig(out, dpi=160, bbox_inches='tight'); plt.close()
    print(f"[ok] {out}")
