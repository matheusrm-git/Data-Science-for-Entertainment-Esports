import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import clear_output
from demoparser2 import DemoParser

SPAWN_COORDINATES = {
    'de_ancient': [(-256.0, 1728.0),
     (-456.0, -2288.0),
     (-512.0, 1696.0),
     (-328.0, -2288.0),
     (-192.0, 1696.0),
     (-448.0, 1728.0),
     (-392.0, -2224.0),
     (-352.0, 1728.0),
     (-584.0, -2288.0),
     (-520.0, -2224.0)],
    'de_anubis': [],
    'de_dust2': [(-493.0, -808.0),
    (-980.0, -754.0),
    (-696.8446044921875, -806.6237182617188),
    (334.3687438964844, 2433.733642578125),
    (-1015.0, -808.0),
    (351.3921203613281, 2352.9423828125),
    (-332.0, -754.0),
    (182.24990844726562, 2439.01171875),
    (160.12274169921875, 2369.67626953125),
    (258.1593933105469, 2480.5537109375)],
    'de_inferno': [(2353.0, 1977.0),
    (2472.349853515625, 2005.969970703125),
    (2456.830078125, 2153.159912109375),
    (-1520.06005859375, 430.8909912109375),
    (-1662.1800537109375, 288.7619934082031),
    (-1675.6199951171875, 351.69500732421875),
    (-1586.52001953125, 440.7900085449219),
    (-1657.22998046875, 419.5769958496094),
    (2397.0, 2079.0),
    (2493.0, 2090.0)],
    'de_mirage': [(-1656.0, -1800.0),
    (-1656.0, -1976.0),
    (-1776.0, -1976.0),
    (1136.0, -256.0),
    (1216.0, -211.0),
    (1136.0, 32.0),
    (-1776.0, -1800.0),
    (1296.0, -352.0),
    (-1720.0, -1896.0),
    (1136.0, -160.0)],
    'de_nuke': [(-1878.0, -980.0),
    (-1832.0, -1160.0),
    (-1929.0, -1025.0),
    (2504.0, -344.0),
    (2552.0, -424.0),
    (2512.0, -504.0),
    (2584.0, -504.0),
    (-1808.0, -1025.0),
    (-1808.0, -1089.0),
    (2585.0, -344.0)],
    'de_overpass': [(-1331.0, -3190.0),
     (-1391.0, -3262.0),
     (-1395.0, -3190.0),
     (-1499.0, -3126.0),
     (-2273.0, 770.0),
     (-2275.0, 842.0),
     (-2199.0, 740.0),
     (-1422.3870849609375, -3129.729248046875),
     (-2190.0, 817.0),
     (-2343.0, 797.0)],
    'de_vertigo': [],
    'de_train': [(1496.0, -1424.0),
    (-2000.0, 1434.233154296875),
    (-1925.0, 1394.0),
    (1462.0, -1226.0),
    (-1850.0, 1256.0),
    (1378.0, -1244.0),
    (1600.0, -1440.0),
    (-1916.0, 1456.233154296875),
    (-1955.0, 1326.0),
    (1542.0, -1336.0)]
    }

FURIA_IDS = {
             'KSCERATO' : 76561198058500492,
             'yuurih' : 76561198164970560,
             'YEKINDAR' : 76561198134401925,
             'FalleN' : 76561197960690195,
             'molodoy' : 76561198200982290
             }

MAP_NAMES = ["de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass","de_train","de_vertigo"]
MAPS_OVERVIEWS = {
     "cs_italy":    (-2647, 2592, 4.6) ,
     "cs_office":   (-1838, 1858, 4.1) ,
     "de_ancient":  (-2953, 2164, 5) ,
     "de_anubis":   (-2796, 3328, 5.22) ,
     "de_dust":     (-2850, 4073, 6),
     "de_dust2":    (-2476, 3239, 4.4) ,
     "de_inferno":  (-2087, 3870, 4.9) ,
     "de_mirage":   (-3230, 1713, 5) ,
     "de_nuke":     (-3453, 2887, 7) ,
     "de_overpass": (-4831, 1781, 5.2) ,
     "de_vertigo":  (-3168, 1762, 4),
     "de_train":    (-2477, 2392, 4.7)
}

MAPS_URL = {
     "de_ancient":    'assets\CS2_maps_radar\de_ancient_radar_psd.png' ,
     "de_anubis":     'assets\CS2_maps_radar\de_anubis_radar_psd.png' ,
     "de_dust2":      'assets\CS2_maps_radar\de_dust2_radar_psd.png' ,
     "de_inferno":    'assets\CS2_maps_radar\de_inferno_radar_psd.png' ,
     "de_mirage":     'assets\CS2_maps_radar\de_mirage_radar_psd.png' ,
     "de_nuke_lower": 'assets\CS2_maps_radar\de_nuke_lower_radar_psd.png' ,
     "de_nuke":       'assets\CS2_maps_radar\de_nuke_radar_psd.png' ,
     "de_overpass":   'assets\CS2_maps_radar\de_overpass_radar_psd.png',
     "de_train":      'assets\CS2_maps_radar\de_train_radar_psd.png'
}

SPAWN_POS_THRESHOLD = 15
IMAGE_SIZE = 1024
NUKE_Z_THRESHOLD = -560

def get_demo_params(parser):
        params = []
        """
        Get important demo parameters.
        - self.parser: demo self.parser object 
        -[0] self.TICK_RATE: tick rate in ticks per second (constant)
        -[1] self.MATCH_START_TICK: match starting tick (after the first freeze time)
        -[2] self.MATCH_RT_HALF_END_TICK: regular time half end tick
        -[3] self.MATCH_END_TICK: match end tick
        -[4] self.N_ROUNDS: number of rounds
        -[5] self.MAP_NAME: map name

        return:
        - params: list of parameters
        """

        # Setting self.TICK_RATE
        params.append(64)

        # Getting  self.MATCH_START_TICK
        try:
            params.append(parser.parse_event("round_announce_match_start").dropna()['tick'][parser.parse_event("round_announce_match_start").shape[0]-1])
        except:
            print("Params might be incorrect")
            params.append(parser.parse_event("round_start").dropna()['tick'][0])

        # Getting self.MATCH_RT_HALF_END_TICK
        try:
            params.append(parser.parse_event('announce_phase_end')['tick'][0])
        except:
            print("Params might be incorrect")
            params.append(parser.parse_event("round_end").dropna()["tick"][11])

        # Getting self.MATCH_END_TICK
        params.append(parser.parse_event("round_end").dropna()['tick'].iloc[-1])

        # Getting self.N_ROUNDS
        player = parser.parse_player_info()['steamid'][0]
        params.append(parser.parse_ticks(wanted_props=['total_rounds_played'], players = [player])['total_rounds_played'].unique()[-1])

        # Getting self.MAP_NAME
        params.append(parser.parse_header()['map_name'])

        return params

def normalize_coords(x_game, y_game, pos_x, pos_y, scale, img_size):
        """
        Convert in-game coordinates to minimap pixels.
        - x_game, y_game: in-game coordinates
        - pos_x, pos_y: minimap overview coordinates
        - scale: minimap overview scale
        - img_size: image resolution (ex.: 1024)
        """
        # Trasforming in-game coords to pixels
        x_img = (x_game - pos_x) / scale
        y_img = (pos_y - y_game) / scale
        return pd.concat([x_img, y_img], axis=1)

def get_stats(match_event_df, names=[], side='CT/T'):
    """
    Get match stats summary.
    - match_event_df: dataframe with match important events
    - name: player name
    - side: side to analyze (CT, T or CT/T) (default = CT/T)

    return:
    - stats_df: Dataframe with match stats
    """
    stats_df = pd.DataFrame()

    match_event_df = match_event_df.reset_index().drop('index', axis=1)
    suicides = match_event_df.loc[match_event_df['attacker_name'] == match_event_df['user_name']]
    match_event_df_k = match_event_df
    match_event_df_d = match_event_df

    if side in ["CT", "T"]:
        if side == 'CT':
            match_event_df_d = match_event_df.loc[match_event_df['side'] == 'CT']
            match_event_df_k = match_event_df.loc[match_event_df['side'] == 'T']
        elif side == 'T':
            match_event_df_d = match_event_df.loc[match_event_df['side'] == 'T']
            match_event_df_k = match_event_df.loc[match_event_df['side'] == 'CT']
    elif side != 'CT/T':
        raise ValueError("Side should be in [CT,T]")

    if not names:
        names = set(match_event_df['attacker_name'].unique().tolist() + match_event_df['user_name'].unique().tolist())

    for name in names:
        # Getting kills
        kills = match_event_df_k.loc[match_event_df['attacker_name'] == name]
        suicide = suicides.loc[suicides['attacker_name'] == name]
        n_kills = len(kills) - len(suicide)

        # Getting headshots percentage
        headshots = kills.loc[kills['headshot'] == True]
        n_headshots = len(headshots)

        if n_kills != 0:
            headshot_percentage = round((n_headshots/n_kills)*100,2)
        else:
            headshot_percentage = 0

        # Getting deaths
        deaths = match_event_df_d.loc[match_event_df['user_name'] == name]
        n_deaths = len(deaths)

        # Getting assists
        assists = match_event_df_k.loc[match_event_df['assister_name'] == name]
        n_assists = len(assists)

        # Getting flash assists
        f_assists = assists.loc[(assists['assistedflash'] == True)]
        n_af = len(f_assists)

        # K/D Ratio
        if n_deaths != 0:
            kd_ratio = round(n_kills/n_deaths,2)
        else:
            kd_ratio = n_kills

        stats = {
            'Player': name,
            'Kills': n_kills,
            'Assists': n_assists,
            'Deaths': n_deaths,
            '% HS': headshot_percentage,
            'Flash Assists': n_af,
            'K/D Ratio': kd_ratio
        }

        stats_df = pd.concat([stats_df, pd.DataFrame(stats, index=[0])], axis=0)

        

    return stats_df #.reset_index().drop('index', axis=1)

class Analyzer:

    def __init__(self, demo_url):
        self.parser = DemoParser(demo_url)
        self.TICK_RATE, self.MATCH_START_TICK, self.MATCH_RT_HALF_END_TICK, self.MATCH_END_TICK, self.N_ROUNDS, self.MAP_NAME = get_demo_params(self.parser)

    def get_match_event_df(self, player_coords):
        """
        Get match important events dataframe.
        - self.parser: demo self.parser object
        - player_coords: dataframe with player coordinates every tick

        return:
        - match_event_df: dataframe with match important events
        """
        match_event_df = self.parser.parse_event('player_death')

        #match_event_df = match_event_df.join(player_coords[['tick', 'round']].drop_duplicates().set_index('tick'), on='tick')
        match_event_df = match_event_df.merge(player_coords[['tick', 'round','name','side']].drop_duplicates(), left_on=['tick','user_name'], right_on=['tick','name'] , how='left')
        match_event_df = match_event_df.dropna(subset=['round'])
        match_event_df = match_event_df.loc[match_event_df['tick'] > self.MATCH_START_TICK]
        match_event_df['side_k'] = match_event_df['side'].apply(lambda x: 'T' if x == 'CT' else 'CT')

        return match_event_df[['attacker_name','user_name','headshot','assister_name','assistedflash', 'revenge','tick', 'round','side','side_k']]

    def get_sides(self, player_id=[], team_num=3):

        start_tick = self.MATCH_RT_HALF_END_TICK+10*self.TICK_RATE
        side_1_half = ''
        side_2_half = ''
        stop_tick = 20
        for i in range(stop_tick):
            aux = self.parser.parse_ticks(wanted_props=["active_weapon_name","team_num"], ticks=[(start_tick)+self.TICK_RATE*i], players = player_id)
            if player_id == []:
                aux = aux.loc[aux['team_num'] == team_num]
            if 'USP-S' in list(aux['active_weapon_name'].unique()):
                side_1_half = 'T'
                side_2_half = 'CT'
                break
            elif 'Glock-18' in list(aux['active_weapon_name'].unique()):
                side_1_half = 'CT'
                side_2_half = 'T'
                break
        return (side_1_half, side_2_half)

    def get_spawn_positions(self):
        """
        Get spawn positions.
        - self.parser: demo self.parser object

        return:
        - spawn_positions: list of spawn positions
        """

        # Getting map spawn positions
        print("Getting Spawns Positions")
        spawn_positions = self.parser.parse_ticks(wanted_props=["X","Y"], ticks=[self.self.MATCH_START_TICK])[["X","Y"]]
        print("Done")
        return [(a, b) for a,b in zip(spawn_positions['X'],spawn_positions['Y'])]

    def get_player_coords(self, player_id=[], other_props=[],verbose=0):
        """
        Get player coordinates.
        - self.parser: demo self.parser object
        - player_id: player id
        - spawn_positions: list of spawn positions
        - half: half of the match (1 or 2) (default = 1)

        return:
        - coords_df_without_spawns: dataframe with player coordinates without spawn positions
        """

        if self.MAP_NAME in MAP_NAMES:
            map_overview = MAPS_OVERVIEWS[self.MAP_NAME]
        else:
            raise ValueError("Map should be in MAP_NAMES")

        if other_props:
                if type(other_props) != list:
                    raise TypeError("other_props mustt be a list of strings")
        else:
            for props in other_props:
                if type(props) != str:
                    raise ValueError("other_props mustt be a list of strings")

        wanted_props = ["X","Y","Z","is_alive",'is_bomb_planted','total_rounds_played','team_num'] + other_props
        print("Getting coordinates")
        if player_id == []:
            coords_df = self.parser.parse_ticks(wanted_props=wanted_props)
        else:
            coords_df = self.parser.parse_ticks(wanted_props=wanted_props, players=player_id)

        #print("Getting sides...")
        #sides = self.get_sides(player_id, coords_df['team_num'].unique()[0])

        print("Dropping NA's values...")
        coords_df.dropna(inplace=True)

        print("Tagging by Half...")
        coords_df['half'] = coords_df['tick'].apply(lambda x: 1 if x <= self.MATCH_RT_HALF_END_TICK else 2)
        print("Tagging by Side...")
        coords_df['side'] = coords_df['team_num'].apply(lambda x: 'T' if x== 2.0 else 'CT')
        # coords_df.loc[coords_df['team_num'] == coords_df['team_num'].unique()[0], 'side'] = coords_df['half'].apply(lambda x: sides[1] if x==2 else sides[0] )
        # coords_df.loc[coords_df['team_num'] == coords_df['team_num'].unique()[1], 'side'] = coords_df['half'].apply(lambda x: sides[0] if x==2 else sides[1] )
        print("Tagging by round...")
        coords_df['round'] = coords_df['total_rounds_played']+1
        print("Tagging by Floor...")
        coords_df['floor'] = coords_df['Z'].apply(lambda x: 0 if x < NUKE_Z_THRESHOLD else 1)


        print("Filtering warmup ticks...")
        coords_df = coords_df.loc[coords_df['tick'] >= self.MATCH_START_TICK]
        print("Filtering after game ticks...")
        coords_df = coords_df.loc[coords_df['tick'] <= self.MATCH_END_TICK]

        print("Normalizing...")
        # Normalizing coords
        coords_df = pd.concat([coords_df.drop(['X','Y','Z','steamid','total_rounds_played'], axis=1),normalize_coords(coords_df["X"], coords_df["Y"], map_overview[0], map_overview[1], map_overview[2], IMAGE_SIZE)], axis=1)
        
        print("Done!")

        if not verbose:
            clear_output()

        return coords_df

    def get_round_ticks(self, upper_limit):
        """
        Get round ticks.
        - self.parser: demo self.parser object
        - lower_limit: start round timer count
        - upper_limit: end round timer count

        return:
        - round_ticks: list of tuples of rounds ticks
        """

        round_start_ticks = list(self.parser.parse_event("round_freeze_end").dropna()['tick'])
        round_end_ticks = list(self.parser.parse_event("round_end").dropna()['tick'])

        if len(round_start_ticks) < len(round_end_ticks):
            dif = len(round_end_ticks)-len(round_start_ticks)
            round_end_ticks = round_end_ticks[dif::]
        elif len(round_start_ticks) > len(round_end_ticks):
            dif = len(round_start_ticks)-len(round_end_ticks)
            round_start_ticks = round_start_ticks[dif::]

        lower_limit_ticks  = round_start_ticks

        if upper_limit > 115:
            upper_limit_ticks = round_end_ticks

        upper_limit_ticks = np.add(round_start_ticks, upper_limit*self.TICK_RATE)

        # Returning list of tuples (start, stop) ticks of each round
        round_ticks = [(a, b) for a,b in zip(lower_limit_ticks,upper_limit_ticks)]

        return round_ticks

    def set_heatmap_data(self, player_coords, names=[], round_seconds=False, upper_limit=0, half=0, side='CT/T', bomb_plt=False, round=0,verbose=0):
        """
        Make heatmap coordinates.
        - self.parser: demo self.parser object
        - player_coords: dataframe with player coordinates
        - map: map name
        - lower_limit: start round timer count
        - upper_limit: end round timer count
        - half: half of the match (1 or 2) (default = 1)

        return:
        - heatmap_df: dataframe with heatmap coordinates
        """

        if round_seconds:
            if upper_limit < 0:
                raise ValueError("Round time limiters must be in [0, 115] seconds")
            elif upper_limit > 115:
                raise ValueError("Round time limiters must be in [0, 115] seconds")
        else:
            upper_limit = 155
        
        if names:
            aux = pd.DataFrame()
            for name in names:
                aux = pd.concat([aux,player_coords.loc[player_coords['name'] == name]], axis=0)
            player_coords = aux
        else:
            raise ValueError("Name not found")
        
        if not (round >= 0 and round < self.N_ROUNDS):
            raise ValueError(f"Round must be in [0, {self.N_ROUNDS}]")

        print("Filtering Alive ticks...")
        player_coords = player_coords.loc[player_coords['is_alive'] == True]

        print("Getting round_ticks...")
        round_ticks = self.get_round_ticks(upper_limit)
        print("Filtering round period...")
        heatmap_df = pd.DataFrame()
        # Filtering coordinates only for the round period chosen
        for round_tick in round_ticks:
            heatmap_df = pd.concat([heatmap_df, player_coords.loc[(player_coords['tick'] >= round_tick[0]) & (player_coords['tick'] <= round_tick[1])]], axis=0)
        heatmap_df.columns = player_coords.columns 
        
        if round:
            print("Choosing round...")
            side = 'CT/T'
            half = 0
            heatmap_df = heatmap_df.loc[heatmap_df['round'] == round]
        
        if side in ["CT", "T"]:
            half = 0
            print("Choosing side...")
            if side == 'CT':
                heatmap_df = heatmap_df.loc[heatmap_df['side'] == 'CT']
            elif side == 'T':
                heatmap_df = heatmap_df.loc[heatmap_df['side'] == 'T']
        elif side != 'CT/T':
            raise ValueError("Side should be in [CT,T]")
        
        if half in [1,2]:
            print("Choosing half...")
            if half == 1:
                heatmap_df = heatmap_df.loc[heatmap_df['half'] == 1]
            else:
                heatmap_df = heatmap_df.loc[heatmap_df['half'] == 2]
        elif half != 0:  
            raise ValueError("Half should be in [1,2]")
        
        if bomb_plt:
            try:
                heatmap_df = heatmap_df.loc[heatmap_df['is_bomb_planted'] == True]
            except KeyError:
                print("There's no 'is_bomb_planted' field")

        print("Done!")

        if not verbose:
            clear_output()
        return heatmap_df

    def get_death_coords(self, players_coords, names=[], half=0, side='CT/T',bomb_plt=False, round=0, verbose=0):
        """
        Get player deaths normalized coordinates.
        - player_id: player id
        - map: map name
        - half: half of the match (1 or 2) (default = 1)

        return:
        - deaths: dataframe with player deaths normalized coordinates
        """
        
        if self.MAP_NAME in MAP_NAMES:
            map_overview = MAPS_OVERVIEWS[self.MAP_NAME]
        else:
            raise ValueError("Map should be in self.MAP_NAMES")

        print("Getting deaths coordinates")
        deaths = self.parser.parse_event("player_death", player= ['X', 'Y', 'Z'])
        
        death_aux = pd.DataFrame()
        players_coords_aux = pd.DataFrame()

        for name in names:
            death_aux = pd.concat([death_aux,deaths.loc[deaths['user_name'] == name]], axis=0)
            players_coords_aux = pd.concat([players_coords_aux,players_coords.loc[players_coords['name'] == name]], axis=0)

        players_coords = players_coords_aux
        deaths = death_aux[['user_X','user_Y','user_Z','tick','user_name']]
        deaths = pd.DataFrame(deaths.reset_index().drop('index', axis=1))
        deaths.columns = ['X','Y','Z','death_tick','name']
        deaths['floor'] = deaths['Z'].apply(lambda x: 0 if x < NUKE_Z_THRESHOLD else 1)

        deaths = deaths.join(players_coords[['tick', 'side', 'round', 'is_bomb_planted']].set_index('tick'), on='death_tick')
        
        if round:
            print("Choosing round...")
            side = 'CT/T'
            half = 0
            deaths = deaths.loc[deaths['round'] == round]
        
        if side in ["CT", "T"]:
            half = 0
            print("Choosing side...")
            if side == 'CT':
                deaths = deaths.loc[deaths['side'] == 'CT']
            elif side == 'T':
                deaths = deaths.loc[deaths['side'] == 'T']
        elif side != 'CT/T':
            raise ValueError("Side should be in [CT,T]")

        if half in [1,2]:
            print("Choosing half")
            deaths['half'] = deaths['death_tick'].apply(lambda x: 1 if x <= self.MATCH_RT_HALF_END_TICK else 2)
            if half == 1:
                deaths = deaths.loc[deaths['half'] == 1]
            else:
                deaths = deaths.loc[deaths['half'] == 2]
        elif half == 0:
            deaths['half'] = deaths['death_tick'].apply(lambda x: 1 if x <= self.MATCH_RT_HALF_END_TICK else 2)
        else:
            raise ValueError("Half should be in [1,2]")

        if bomb_plt:
            try:
                deaths = deaths.loc[deaths['is_bomb_planted'] == True]
            except KeyError:
                print("There's no 'is_bomb_planted' field")

        print("Normalizing deaths")
        deaths = pd.concat([deaths['name'],deaths['half'],deaths['floor'],normalize_coords(deaths["X"], deaths["Y"], map_overview[0], map_overview[1], map_overview[2], IMAGE_SIZE)], axis=1) 
        print("Done!")

        if not verbose:
            clear_output()

        return deaths

    def get_kills_coords(self, players_coords, names=[], half=0, side='CT/T',bomb_plt=False, round=0, verbose=0):
        """
        Get player kills normalized coordinates.
        - player_id: player id
        - map: map name
        - half: half of the match (1 or 2) (default = 1)

        return:
        - kills: dataframe with player kills normalized coordinates
        """
        
        if self.MAP_NAME in MAP_NAMES:
            map_overview = MAPS_OVERVIEWS[self.MAP_NAME]
        else:
            raise ValueError("Map should be in self.MAP_NAMES")

        print("Getting kills coordinates")
        kills = self.parser.parse_event("player_death", player= ['X', 'Y', 'Z'])
        
        kills_aux = pd.DataFrame()
        players_coords_aux = pd.DataFrame()
        for name in names:
            kills_aux = pd.concat([kills_aux,kills.loc[kills['attacker_name'] == name]], axis=0)
            players_coords_aux = pd.concat([players_coords_aux,players_coords.loc[players_coords['name'] == name]], axis=0)

        players_coords = players_coords_aux
        kills = kills_aux[['user_X','user_Y','user_Z','tick','attacker_name']]
        kills = pd.DataFrame(kills.reset_index().drop('index', axis=1))
        kills.columns = ['X','Y','Z','kill_tick','name']
        kills['floor'] = kills['Z'].apply(lambda x: 0 if x < NUKE_Z_THRESHOLD else 1)

        kills = kills.join(players_coords[['tick', 'side', 'round', 'is_bomb_planted']].set_index('tick'), on='kill_tick')
        
        if round:
            print("Choosing round...")
            side = 'CT/T'
            half = 0
            kills = kills.loc[kills['round'] == round]
        
        if side in ["CT", "T"]:
            half = 0
            print("Choosing side...")
            if side == 'CT':
                kills = kills.loc[kills['side'] == 'CT']
            elif side == 'T':
                kills = kills.loc[kills['side'] == 'T']
        elif side != 'CT/T':
            raise ValueError("Side should be in [CT,T]")

        if half in [1,2]:
            print("Choosing half")
            kills['half'] = kills['kill_tick'].apply(lambda x: 1 if x <= self.MATCH_RT_HALF_END_TICK else 2)
            if half == 1:
                kills = kills.loc[kills['half'] == 1]
            else:
                kills = kills.loc[kills['half'] == 2]
        elif half == 0:
            kills['half'] = kills['kill_tick'].apply(lambda x: 1 if x <= self.MATCH_RT_HALF_END_TICK else 2)
        else:
            raise ValueError("Half should be in [1,2]")

        if bomb_plt:
            try:
                kills = kills.loc[kills['is_bomb_planted'] == True]
            except KeyError:
                print("There's no 'is_bomb_planted' field")

        print("Normalizing kills")
        kills = pd.concat([kills['name'],kills['half'],kills['floor'],normalize_coords(kills["X"], kills["Y"], map_overview[0], map_overview[1], map_overview[2], IMAGE_SIZE)], axis=1) 
        print("Done!")

        if not verbose:
            clear_output()

        return kills

    def generate_map_image(self, title=''):
        
        print('Generating map image...')
        if self.MAP_NAME == 'de_nuke':
            map_image = mpimg.imread(MAPS_URL['de_nuke'])
            map_image_lower = mpimg.imread(MAPS_URL['de_nuke_lower'])
            fig = plt.figure(figsize=(32,16))
            if title != '':
                plt.title(title)
            plt.axis('off')
            plt.subplot(1, 2, 1)
            plt.imshow(map_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE,0])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(map_image_lower, extent=[0, IMAGE_SIZE, IMAGE_SIZE,0])
            plt.axis('off')
        else:
            map_image = mpimg.imread(MAPS_URL[self.MAP_NAME])

            fig = plt.figure(figsize=(16,16))
            if title != '':
                plt.title(title)
            plt.imshow(map_image, extent=[0, IMAGE_SIZE, IMAGE_SIZE,0])
            plt.axis('off')
        return fig

    def generate_heatmap(self, heatmap_df):
        """
        Generate heat map.
        - map_image: map image
        - heatmap_df: dataframe with heatmap coordinates
        """
        
        cmap = sns.color_palette("YlOrBr", as_cmap=True)

        print('Generating heatmap...')

        if self.MAP_NAME == 'de_nuke':
            heatmap_df_higher = heatmap_df.loc[heatmap_df['floor'] == 1]
            heatmap_df_lower = heatmap_df.loc[heatmap_df['floor'] == 0]

            plt.subplot(1, 2, 1)
            sns.kdeplot(x=heatmap_df_higher["X"], y=heatmap_df_higher["Y"], fill=True, alpha=0.2,thresh=0.05, levels=100, cmap=cmap)
            plt.axis('off')


            plt.subplot(1, 2, 2)
            sns.kdeplot(x=heatmap_df_lower["X"], y=heatmap_df_lower["Y"], fill=True, alpha=0.2,thresh=0.05, levels=100, cmap=cmap)
            plt.axis('off')
        
        else:
            # Generating heatmap
            sns.kdeplot(x=heatmap_df["X"], y=heatmap_df["Y"], fill=True, alpha=0.2,thresh=0.05, levels=100, cmap=cmap)
            plt.axis('off')

    def generate_death_marks(self, death_marks=()):
        """
        Generate heat map.
        - map_image: map image
        - heatmap_coords: dataframe with heatmap coordinates
        """
        cmap = sns.color_palette("YlOrBr", as_cmap=True)

        print('Generating death marks...')

        if self.MAP_NAME == 'de_nuke':
            deaths_coords_higher = death_marks.loc[death_marks['floor'] == 1]
            deaths_coords_lower = death_marks.loc[death_marks['floor'] == 0]

            plt.subplot(1, 2, 1)
            plt.plot(deaths_coords_higher['X'], deaths_coords_higher['Y'], 'x', markersize=15, color='r')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(deaths_coords_lower['X'], deaths_coords_lower['Y'], 'x', markersize=15, color='r')
            plt.axis('off')
        
        else:
            # Generating heatmap
            plt.plot(death_marks['X'], death_marks['Y'], 'x', markersize=15, color='r')
            plt.axis('off')
        
    def generate_kill_marks(self, kill_marks=()):
        """
        Generate heat map.
        - map_image: map image
        - heatmap_coords: dataframe with heatmap coordinates
        """
        cmap = sns.color_palette("YlOrBr", as_cmap=True)

        print('Generating kill marks...')

        if self.MAP_NAME == 'de_nuke':
            kill_coords_higher = kill_marks.loc[kill_marks['floor'] == 1]
            kill_coords_lower = kill_marks.loc[kill_marks['floor'] == 0]

            plt.subplot(1, 2, 1)
            plt.plot(kill_coords_higher['X'], kill_coords_higher['Y'], 'x', markersize=15, color='g')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(kill_coords_lower['X'], kill_coords_lower['Y'], 'x', markersize=15, color='g')
            plt.axis('off')
        
        else:
            # Generating heatmap
            plt.plot(kill_marks['X'], kill_marks['Y'], 'x', markersize=15, color='g')
            plt.axis('off')

    def map_graph_analysis(self,names=[],round_seconds=False,upper_limit=0, half=0, side='CT/T',heatmap=True, deaths=False, kills=0, bomb_plt=False, round=0,verbose=0):

        # Adjusting heatmap params
        player_coords = self.get_player_coords(verbose=verbose)

        if names == []:
            input_names = [n for n in player_coords['name'].unique()]

        else:
            input_names = names
        heatmap_df = self.set_heatmap_data(player_coords,names=input_names,round_seconds=round_seconds, upper_limit=upper_limit,half=half, side=side,bomb_plt=bomb_plt, round=round,verbose=verbose)
    
        if not heatmap_df.empty:
            if half != 0 and side == 'CT/T':
                side = list(heatmap_df['side'].unique())[0]
            elif half == 0 and side != 'CT/T':
                half = list(heatmap_df['half'].unique())[0]
        else:
            raise ValueError("Player name(s) probably incorrect")

        if deaths:
            death_coords = self.get_death_coords(player_coords, input_names, half, side=side, bomb_plt=bomb_plt, round=round,verbose=verbose)
        else:
            death_coords = ()

        if kills:
            kill_coords = self.get_kills_coords(player_coords, input_names, half, side=side, bomb_plt=bomb_plt, round=round)
        else:
            kill_coords = ()
    
        half_cardinality = {0 : 'Full Game', 1 : 'First', 2 : 'Second'}
        title = f"Heatmap for {side} side ({self.MAP_NAME}, {half_cardinality[half]} Half)"
    
        # Generating map image
        fig = self.generate_map_image(title)

        # Generating features
        if heatmap:
            self.generate_heatmap(heatmap_df)
        if deaths:
            self.generate_death_marks(death_coords)
        if kills:
            self.generate_kill_marks(kill_coords)

        # Cleaning the output terminal
        clear_output()

    def map_stats_analysis(self, names=[], side='CT/T'):
        """
        Get match stats summary.
        - match_event_df: dataframe with match important events
        - name: player name
        - side: side to analyze (CT, T or CT/T) (default = CT/T)

        return:
        - stats_df: Dataframe with match stats
        """

        match_event_df = self.get_match_event_df(self.get_player_coords())

        if names == []:
            input_names = list(set([n for n in match_event_df['attacker_name'].unique().tolist() + match_event_df['user_name'].unique().tolist()]))
        else:
            input_names = names

        stats_df = get_stats(match_event_df, input_names, side)

        if not stats_df.empty:
            clear_output()
            return stats_df.reset_index().sort_values(by='K/D Ratio', ascending=False).drop('index', axis=1)
        else:
            raise ValueError("Player name(s) probably incorrect")
        
    def generate_dashboard_graph_analysis(self,player_coords,names=[],round_seconds=False,upper_limit=0, half=0, side='CT/T',heatmap=True, deaths=False, kills=0, bomb_plt=False, round=0,verbose=0):

        # Adjusting heatmap params

        if names == []:
            input_names = [n for n in player_coords['name'].unique()]

        else:
            input_names = names
        heatmap_df = self.set_heatmap_data(player_coords,names=input_names,round_seconds=round_seconds, upper_limit=upper_limit,half=half, side=side,bomb_plt=bomb_plt, round=round,verbose=verbose)
    
        if not heatmap_df.empty:
            if half != 0 and side == 'CT/T':
                side = list(heatmap_df['side'].unique())[0]
            elif half == 0 and side != 'CT/T':
                half = list(heatmap_df['half'].unique())[0]
        else:
            raise ValueError("Player name(s) probably incorrect")

        if deaths:
            death_coords = self.get_death_coords(player_coords, input_names, half, side=side, bomb_plt=bomb_plt, round=round,verbose=verbose)
        else:
            death_coords = ()

        if kills:
            kill_coords = self.get_kills_coords(player_coords, input_names, half, side=side, bomb_plt=bomb_plt, round=round)
        else:
            kill_coords = ()
    
        half_cardinality = {0 : 'Full Game', 1 : 'First', 2 : 'Second'}
    
        # Cleaning the output terminal
        clear_output()
    
        # Generating map image
        fig = self.generate_map_image()

        # Generating features
        if heatmap:
            self.generate_heatmap(heatmap_df)
        if deaths:
            self.generate_death_marks(death_coords)
        if kills:
            self.generate_kill_marks(kill_coords)

        return fig