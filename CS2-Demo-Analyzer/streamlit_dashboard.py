import streamlit as st
import demo_analyzer as da
from threading import RLock
import gdown

# Caching functions
@st.cache_data
def load_demo(demo_url, desired_path):
    gdown.download(demo_url, desired_path, quiet=False, fuzzy=True)

@st.cache_data
def load_player_coords():
    return analyzer.get_player_coords()

@st.cache_data
def load_players_stats():
    return analyzer.map_stats_analysis()

# Initializing a lock for thread safety
_lock = RLock()

# Downloading demo file
demo_url = 'https://drive.google.com/file/d/1GrL35_crPml7vnM6xkDYpcUA6G-sCOnK/view?usp=sharing'
demo_path = "furia-vs-legacy-m3-mirage.dem"
load_demo(demo_url, demo_path)

# Initializing analyzer and loading data
analyzer = da.Analyzer(demo_path)
players_coords = load_player_coords()
players_stats = load_players_stats()

# Streamlit App
st.set_page_config(page_title="Demo Analyzer Dashboard")

st.title("CS2 Demo Analyzer Dashboard")

# Defining tabs
tab_1, tab_2 = st.tabs(["Match Overall","Player Analysis"])

# Tab 1: Match Overall Statistics
with tab_1:
    st.image("CS2-Demo-Analyzer/assets/header_images/legacy_vs_furia_blast_open_london_2025.png")
    st.header("Match Overall Statistics")
    st.dataframe(players_stats, hide_index=True)

# Tab 2: Player Analysis
with tab_2:
    st.header("Player Analysis")
    
    # Filters for player analysis (player name, side, round seconds, features to display)
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
            bomb_plt = st.checkbox("After plant", value=False)

    # Analyze button
    if st.button("Analyze"):
        # Validating inputs
        if not (name[0] in player_names and side in sides):
            st.error("Please select a valid player and side.")
        else:
            with st.spinner('Analyzing Statistics...'):
                players_stats = analyzer.map_stats_analysis(names=name, side=side)

            # Displaying player stats (player image, kills, deaths, assists, K/D ratio, flash assists, HS%)
            stats = st.container()
            with stats:
                # Define two columns for player image and stats (stats_col1 for player image, stats_col2 for player stats)
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    player_img = f"CS2-Demo-Analyzer/assets/players_images/{selected_player}.png"
                    st.image(player_img)
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
            
            # Generating and displaying the graph    
            map = st.container()
            with map:
                with st.spinner('Generating Graph...'):
                    with _lock:
                        fig = analyzer.generate_dashboard_graph_analysis(players_coords, names=name,round_seconds=round_seconds,upper_limit=upper_limit, side=side,heatmap=heatmap, deaths=deaths,kills=kills,bomb_plt=bomb_plt)
                st.pyplot(fig)
    
        

