import json, numpy as np
from gudhi import CubicalComplex
from scipy.ndimage import distance_transform_edt

def compute_ph_skeleton(pom_npz: str, out_json: str, max_features=64):
    data = np.load(pom_npz)
    prob = data["prob"]
    img = 1.0 - prob
    cc = CubicalComplex(top_dimensional_cells=img.astype(float))
    pd = cc.persistence()
    H0 = [(b, d) for (dim, (b, d)) in pd if dim == 0 and np.isfinite(d)]
    H1 = [(b, d) for (dim, (b, d)) in pd if dim == 1 and np.isfinite(d)]
    Dt = distance_transform_edt(prob < 0.5)
    out = {
        "H0_count": len(H0),
        "H1_count": len(H1),
        "H0_top": sorted([{"birth": float(b), "death": float(d), "pers": float(d-b)} for b,d in H0], key=lambda x:-x["pers"])[:max_features],
        "H1_top": sorted([{"birth": float(b), "death": float(d), "pers": float(d-b)} for b,d in H1], key=lambda x:-x["pers"])[:max_features],
        "skeleton_hint": {"shape": list(Dt.shape)}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ok] PH skeleton -> {out_json}")
