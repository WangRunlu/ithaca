import argparse, os
from data.ithaca365_paths import ensure_dirs, default_download_targets, ensure_devkit_ready
from data.download_subset import download_subset
from data.extract_all import extract_all
from mapping.occupancy_map import build_pom
from mapping.persistent_homology import compute_ph_skeleton
from topology.dance_anchors import pick_dance_anchors_and_dense_samples
from topology.gng_graph import run_gng_build
from planning.compute_weights import compute_edge_weights
from planning.path_planning import build_candidates_and_dp_teacher
from rl_agent.bc_pretrain import bc_pretrain_from_teacher
from rl_agent.sac_training import train_sac_agent
from visualization.plot_map import plot_bev_maps
from visualization.plot_trajectories import plot_teacher_student

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, default=os.environ.get("DATAROOT","/data/sets/ithaca365"))
    p.add_argument("--workdir", type=str, default="outputs")
    p.add_argument("--use_subset", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=400)
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    ensure_dirs(args.dataroot, args.workdir)
    ensure_devkit_ready()

    if int(args.use_subset)==1:
        targets = default_download_targets()
        download_subset(args.dataroot, targets)
        extract_all(args.dataroot)

    pom_npz = os.path.join(args.workdir, "pom_map.npz")
    meta_json = os.path.join(args.workdir, "frame_index.json")
    build_pom(args.dataroot, pom_npz, meta_json, max_frames=args.max_frames)

    ph_json = os.path.join(args.workdir, "ph_skeleton.json")
    compute_ph_skeleton(pom_npz, ph_json)

    anchors_json = os.path.join(args.workdir, "dance_anchors.json")
    dense_pts_npz = os.path.join(args.workdir, "dense_samples.npz")
    pick_dance_anchors_and_dense_samples(pom_npz, ph_json, anchors_json, dense_pts_npz)

    gng_graph_npz = os.path.join(args.workdir, "gng_graph.npz")
    run_gng_build(dense_pts_npz, gng_graph_npz)

    edge_attr_json = os.path.join(args.workdir, "edge_attr.json")
    compute_edge_weights(args.dataroot, gng_graph_npz, meta_json, pom_npz, edge_attr_json)

    teacher_npz = os.path.join(args.workdir, "teacher_traj.npz")
    build_candidates_and_dp_teacher(gng_graph_npz, edge_attr_json, teacher_npz)

    agent_dir = os.path.join(args.workdir, "agent"); os.makedirs(agent_dir, exist_ok=True)
    bc_pretrain_from_teacher(teacher_npz, agent_dir)
    policy_path = train_sac_agent(gng_graph_npz, edge_attr_json, teacher_npz, agent_dir)

    plot_bev_maps(pom_npz, ph_json, gng_graph_npz, out_dir=args.workdir)
    plot_teacher_student(gng_graph_npz, teacher_npz, policy_path, out_dir=args.workdir)
    print("\n✅ Done. 输出目录：", os.path.abspath(args.workdir))

if __name__ == "__main__":
    main()
