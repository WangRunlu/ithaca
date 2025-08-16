import os, zipfile, tarfile

def _safe_extract_zip(zpath, out_dir):
    with zipfile.ZipFile(zpath, 'r') as zf:
        for m in zf.namelist():
            if m.endswith('/'): continue
            target = os.path.join(out_dir, m)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                zf.extract(m, out_dir)

def _safe_extract_tar(tpath, out_dir):
    with tarfile.open(tpath, 'r:gz') as tf:
        for m in tf.getmembers():
            if m.isdir(): continue
            target = os.path.join(out_dir, m.name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                tf.extract(m, out_dir)

def extract_all(dataroot: str):
    dldir = os.path.join(dataroot, "_downloads")
    sub_zip = os.path.join(dldir, "Ithaca365-sub.zip")
    tables_tar = os.path.join(dldir, "v2.21.tar.gz")
    road_zip = os.path.join(dldir, "Ithaca365_road.zip")
    road_split_zip = os.path.join(dldir, "Ithaca365_road_split.zip")
    if os.path.exists(sub_zip): _safe_extract_zip(sub_zip, dataroot)
    if os.path.exists(tables_tar): _safe_extract_tar(tables_tar, dataroot)
    if os.path.exists(road_zip): _safe_extract_zip(road_zip, dataroot)
    if os.path.exists(road_split_zip): _safe_extract_zip(road_split_zip, dataroot)
    print("[ok] 解压完成（samples/、sweeps/、v1.0-*）。")
