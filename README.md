# Ithaca365 HM→POM→PH→DANCE→GNG→DP→SAC (Subset, Python + Docker)

用途：下载 Ithaca365 子集，跑完整流水线（POM/PH/DANCE/GNG/权重/k最短路+DP/SAC/可视化）。
许可：CC BY‑NC‑SA 4.0（非商业）。依赖 ithaca365-devkit。

## Docker 快速开始
docker build -t ithaca365-pipe:latest .
docker run --rm -it -v /PATH/TO/DATASETS:/data/sets -v $(pwd):/workspace ithaca365-pipe:latest bash
python3 run_all.py --dataroot /data/sets/ithaca365 --use_subset 1 --max_frames 400
