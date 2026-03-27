import streamlit as st

def main():

    pg = st.navigation([st.Page("pages/01_LaunchPage.py", title="Launchpage", icon=":material/home:"), st.Page("pages/02_Base_Sightings.py", title="Inspect vehicle sightings", icon=":material/search:"), st.Page("pages/03_Vehicle_Events.py", title="ReID'd vehicle drilldown", icon=":material/search:"), st.Page("pages/04_ReID_timeline.py", title="ReID timeline", icon=":material/search:"), st.Page("pages/User_UI/01_Traffic_Overview.py", title="Traffic Overview", icon=":material/search:"), st.Page("pages/User_UI/02_Transitions_Detailed.py", title="Camera transitions detailed", icon=":material/search:"), st.Page("pages/User_UI/03_Exit_Analysis.py", title="Camera exit analysis", icon=":material/search:"), st.Page("pages/User_UI/04_Camera_Utilization.py", title="Camera Utilization", icon=":material/search:"), st.Page("pages/User_UI/05_Camera_Flow_Graph.py", title="Camera Flow graph", icon=":material/search:"), st.Page("pages/User_UI/06_ReID_Log.py", title="Re-identification log", icon=":material/search:")])
    pg.run()


if __name__ == "__main__":
    main()

    