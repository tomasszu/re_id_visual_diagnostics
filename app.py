import streamlit as st
import argparse

# ---------- args ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio_endpoint', type=str, default='localhost:9000')
    parser.add_argument('--minio_access_key', type=str, default="minioadmin")
    parser.add_argument('--minio_secret_key', type=str, default="minioadmin")
    parser.add_argument('--minio_bucket', type=str, default="reid-service")
    parser.add_argument('--minio_secure', action='store_true', help='Use secure HTTPS MinIO connection')

    # important for streamlit compatibility
    args, _ = parser.parse_known_args()
    return args


def init_runtime_config():
    if "runtime_config" not in st.session_state:
        args = parse_args()

        st.session_state["runtime_config"] = {
            "MINIO_ENDPOINT": args.minio_endpoint,
            "MINIO_ACCESS_KEY": args.minio_access_key,
            "MINIO_SECRET_KEY": args.minio_secret_key,
            "MINIO_BUCKET": args.minio_bucket,
            "MINIO_SECURE": args.minio_secure,
        }


def main():
    init_runtime_config()

    pg = st.navigation([
        st.Page("pages/01_LaunchPage.py", title="Launchpage", icon=":material/home:"),
        st.Page("pages/02_Base_Sightings.py", title="Inspect vehicle sightings", icon=":material/search:"),
        st.Page("pages/03_Vehicle_Events.py", title="ReID'd vehicle drilldown", icon=":material/search:"),
        st.Page("pages/04_ReID_timeline.py", title="ReID timeline", icon=":material/search:"),
        st.Page("pages/User_UI/01_Traffic_Overview.py", title="Traffic Overview", icon=":material/search:"),
        st.Page("pages/User_UI/02_Transitions_Detailed.py", title="Camera transitions detailed", icon=":material/search:"),
        st.Page("pages/User_UI/03_Exit_Analysis.py", title="Camera exit analysis", icon=":material/search:"),
        st.Page("pages/User_UI/04_Camera_Utilization.py", title="Camera Utilization", icon=":material/search:"),
        st.Page("pages/User_UI/05_Camera_Flow_Graph.py", title="Camera Flow graph", icon=":material/search:"),
        st.Page("pages/User_UI/06_ReID_Log.py", title="Re-identification log", icon=":material/search:"),
        st.Page("pages/gt_creation/01_ground_truth_assignment_lite.py", title="Ground Truth Assignment Visual (Lite)", icon=":material/assignment:"),
        st.Page("pages/gt_creation/02_ground_truth_assignment_full.py", title="Ground Truth Assignment Table (Full)", icon=":material/assignment:"),
        st.Page("pages/gt_creation/03_gt_consistency.py", title="GT Consistency", icon=":material/sync:"),
        st.Page("pages/gt_creation/04_gt_merge.py", title="GT Merge", icon=":material/merge:")
    ])
    pg.run()


if __name__ == "__main__":
    main()

    