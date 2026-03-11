import streamlit as st

def main():

    pg = st.navigation([st.Page("pages/01_LaunchPage.py", title="Launchpage", icon=":material/home:"), st.Page("pages/02_Base_Sightings.py", title="Inspect vehicle sightings", icon=":material/search:"), st.Page("pages/03_Analysed_Sightings.py", title="Inspect analysed vehicle sightings", icon=":material/search:"), st.Page("pages/04_Vehicle_Events.py", title="Inspect and merge vehicle events", icon=":material/search:")])
    pg.run()


if __name__ == "__main__":
    main()

    