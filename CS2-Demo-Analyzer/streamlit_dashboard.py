import streamlit as st
import demo_analyzer as da
from threading import RLock
import gdown
from pathlib import Path

_lock = RLock()

current_dir = Path(__file__).parent

demo_url = 'https://drive.google.com/file/d/1GrL35_crPml7vnM6xkDYpcUA6G-sCOnK/view?usp=sharing'
path = "furia-vs-legacy-m3-mirage.dem"

@st.cache_data
def load_demo():
    gdown.download(demo_url, path, quiet=False, fuzzy=True)

@st.cache_data
def load_player_coords():
    return analyzer.get_player_coords()

@st.cache_data
def load_players_stats():
    return analyzer.map_stats_analysis()

load_demo()
analyzer = da.Analyzer(path)
players_coords = load_player_coords()
players_stats = load_players_stats()

st.title("Demo Analyzer Dashboard")

tab_1, tab_2 = st.tabs(["Match Overall","Player Analysis"])

with tab_1:
    st.image(current_dir / "/assets/header_images/legacy_vs_furia_blast_open_london_2025.png")
    st.header("Match Overall Statistics")
    st.dataframe(players_stats, hide_index=True)

with tab_2:
    st.header("Player Analysis")
    filters = st.container()
    with filters:

        f_c1, f_c2 = st.columns(2)
        with f_c1:
            player_names = ['FalleN', 'KSCERATO', 'yuurih', 'molodoy', 'YEKINDAR']
            selected_player = st.selectbox("Select Player", player_names)
            name = [selected_player]  
            round_seconds = st.toggle("Filter Round Seconds", value=False)
        with f_c2:
            sides = ['CT', 'T', 'CT/T']
            side = st.selectbox("Select Side", sides)   
            if round_seconds:
                upper_limit = st.slider(label='Seconds',min_value=0, max_value=115, value=115, step=1)
            else:
                upper_limit = 0
        
        check_boxes = filters.container()
        b1, b2, b3, b4 = check_boxes.columns(4)
        with b1:
            heatmap = st.checkbox("Positions Heatmap", value=True)
        with b2:
            deaths = st.checkbox("Deaths", value=True)
        with b3:
            kills = st.checkbox("Kills", value=True)
        with b4:
            bomb_plt = st.checkbox("Afterplant", value=False)


    if st.button("Analyze"):
        if not (name[0] in player_names and side in sides):
            st.error("Please select a valid player and side.")
        else:
            with st.spinner('Analyzing Statistics...'):
                players_stats = analyzer.map_stats_analysis(names=name, side=side)

            stats = st.container()
            with stats:
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.image(current_dir / f"/assets/players_players/{selected_player}.png")
                with stats_col2:
                    st.subheader(f"{selected_player}")
                    stats_cb1, stats_cb2,stats_cb3 = stats_col2.columns(3)
                    with stats_cb1:
                        st.metric("Kills", int(players_stats.loc[players_stats['Player'] == selected_player]['Kills']))
                        st.metric("Assists", int(players_stats.loc[players_stats['Player'] == selected_player]['Assists'])) 
                    with stats_cb2:
                        st.metric("Deaths", int(players_stats.loc[players_stats['Player'] == selected_player]['Deaths']))
                        st.metric("K/D Ratio", round(float(players_stats.loc[players_stats['Player'] == selected_player]['K/D Ratio']),2))
                    with stats_cb3:
                        st.metric("Flash Assists", round(float(players_stats.loc[players_stats['Player'] == selected_player]['Flash Assists']),2))
                        st.metric("% HS", str(round(float(players_stats.loc[players_stats['Player'] == selected_player]['% HS']),1)) + "%")
                
            map = st.container()
            with map:
                with st.spinner('Generating Graph...'):
                    with _lock:
                        fig = analyzer.generate_dashboard_graph_analysis(players_coords, names=name,round_seconds=round_seconds,upper_limit=upper_limit, side=side,heatmap=heatmap, deaths=deaths,kills=kills,bomb_plt=bomb_plt)
                st.pyplot(fig)
    
        

