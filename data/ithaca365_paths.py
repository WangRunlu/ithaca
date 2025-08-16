import os

def ensure_dirs(dataroot: str, workdir: str):
    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(os.path.join(dataroot, "_downloads"), exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

def ensure_devkit_ready():
    try:
        import ithaca365  # noqa: F401
    except Exception as e:
        raise RuntimeError("未检测到 ithaca365-devkit。请在 Docker 容器内运行或 pip 安装 devkit。") from e

def default_download_targets():
    return {
        "subset_zip": "http://en-ma-sdc.coecis.cornell.edu/web_data/Ithaca365-sub.zip",
        "tables_tar": "http://en-ma-sdc.coecis.cornell.edu/web_data/v2.21.tar.gz",
        "road_zip": "http://en-ma-sdc.coecis.cornell.edu/web_data/Ithaca365_road.zip",
        "road_split_zip": "http://en-ma-sdc.coecis.cornell.edu/web_data/Ithaca365_road_split.zip"
    }
