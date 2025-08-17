import os, json, numpy as np
from tqdm import tqdm

def _setup_grid(xmin=-200, xmax=200, ymin=-200, ymax=200, res=0.5):
    xs = np.arange(xmin, xmax, res)
    ys = np.arange(ymin, ymax, res)
    return xs, ys, np.zeros((len(ys), len(xs)), dtype=np.float32)

def _logit(p):
    p = np.clip(p, 1e-4, 1-1e-4)
    return np.log(p/(1-p))

def _inv_logit(l):
    return 1/(1+np.exp(-l))

def _update_hits(gridL, xs, ys, pts_xy, p_hit=0.7):
    H, W = gridL.shape
    lp_hit = _logit(p_hit)
    dx = xs[1]-xs[0]; dy = ys[1]-ys[0]
    for x, y in pts_xy:
        ix = int((x - xs[0]) / dx); iy = int((y - ys[0]) / dy)
        if 0 <= ix < W and 0 <= iy < H:
            gridL[iy, ix] += lp_hit

def build_pom(dataroot: str, out_npz: str, out_index_json: str, max_frames=400):
    from ithaca365.ithaca365 import Ithaca365
    nusc = Ithaca365(dataroot=dataroot, version='v1.0-trainval', verbose=True)

    xs, ys, gridL = _setup_grid()
    frames = []; cnt = 0
    for sample in tqdm(nusc.sample, desc="POM"):
        if max_frames and cnt >= max_frames: break
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if lidar_token is None: continue
        sd_rec = nusc.get('sample_data', lidar_token)
        lidar_path = nusc.get_sample_data_path(lidar_token)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(lidar_path)
            pts = np.asarray(pcd.points, dtype=np.float32)
        except Exception:
            pts = np.fromfile(lidar_path, dtype=np.float32)
            D = 4 if pts.size % 5 else 5
            pts = pts.reshape(-1, D)[:, :3]
        _update_hits(gridL, xs, ys, pts[:, :2])
        frames.append({
            "timestamp": sample.get("timestamp", 0),
            "scene": sample.get("scene_token", ""),
            "lidar_path": lidar_path,
        })
        cnt += 1

    prob = _inv_logit(gridL)
    np.savez_compressed(out_npz, xs=xs, ys=ys, logodds=gridL, prob=prob)
    with open(out_index_json, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, ensure_ascii=False, indent=2)
    print(f"[ok] POM -> {out_npz}, 索引 -> {out_index_json}")
