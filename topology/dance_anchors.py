import json, numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.feature import peak_local_max

def pick_dance_anchors_and_dense_samples(pom_npz: str, ph_json: str, out_anchors_json: str, out_dense_npz: str,
                                         top_k=32, radius=8, dense_per_anchor=200):
    data = np.load(pom_npz)
    xs, ys, prob = data["xs"], data["ys"], data["prob"]
    free = (prob < 0.45).astype(np.float32)
    Dt = distance_transform_edt(free > 0.5)
    score = gaussian_filter(Dt, sigma=2.0)
    peaks = peak_local_max(score, min_distance=radius, threshold_rel=0.2)
    peaks = sorted(peaks, key=lambda ij: score[ij[0], ij[1]], reverse=True)[:top_k]
    anchors = []
    for ij in peaks:
        i, j = int(ij[0]), int(ij[1])
        anchors.append({"iy": i, "ix": j, "x": float(xs[j]), "y": float(ys[i]), "score": float(score[i,j])})
    pts = []
    for a in anchors:
        i0, j0 = a["iy"], a["ix"]
        for _ in range(dense_per_anchor):
            di = np.random.randint(-radius, radius+1)
            dj = np.random.randint(-radius, radius+1)
            i = np.clip(i0+di, 0, free.shape[0]-1)
            j = np.clip(j0+dj, 0, free.shape[1]-1)
            if free[i,j] > 0.5:
                pts.append((xs[j], ys[i]))
    pts = np.array(pts, dtype=np.float32)
    with open(out_anchors_json, "w", encoding="utf-8") as f:
        json.dump({"anchors": anchors}, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dense_npz, pts=pts)
    print(f"[ok] DANCE anchors -> {out_anchors_json}, dense samples -> {out_dense_npz} ({len(pts)} pts)")
