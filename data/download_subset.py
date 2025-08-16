import os, urllib.request
from .ithaca365_paths import default_download_targets

def _download(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] 已存在: {out_path}")
        return
    print(f"[wget] {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[ok] {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

def download_subset(dataroot: str, targets=None):
    if targets is None:
        targets = default_download_targets()
    dldir = os.path.join(dataroot, "_downloads")
    os.makedirs(dldir, exist_ok=True)
    _download(targets["subset_zip"], os.path.join(dldir, "Ithaca365-sub.zip"))
    _download(targets["tables_tar"], os.path.join(dldir, "v2.21.tar.gz"))
    _download(targets["road_zip"], os.path.join(dldir, "Ithaca365_road.zip"))
    _download(targets["road_split_zip"], os.path.join(dldir, "Ithaca365_road_split.zip"))
    print("[done] 下载完成。")
