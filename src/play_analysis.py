import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

class PlayPhase(Enum):
    PRE_SNAP = "pre_snap"
    POST_SNAP = "post_snap"
    PASS_DECISION = "pass_decision"
    PASS_ARRIVAL = "pass_arrival"
    PLAY_END = "play_end"

@dataclass
class PlayEvent:
    frame_id: int
    event_type: str
    phase: PlayPhase


class PlayAnalyzer:
    def __init__(self, 
                 df_tracking: pd.DataFrame,
                 df_players: pd.DataFrame,
                 df_plays: pd.DataFrame,
                 df_games: pd.DataFrame,
                 df_player_plays: pd.DataFrame):
        self.df_tracking = df_tracking
        self.df_players = df_players
        self.df_plays = df_plays
        self.df_games = df_games
        self.df_player_plays = df_player_plays
        
        # Define key events for different phases
        self.pre_snap_events = ['huddle_start_offense','huddle_break_offense', 'line_set', 'man_in_motion', 'shift']
        self.snap_events = ['ball_snap', 'snap_direct']
        self.pass_events = ['pass_forward','pass_shovel', 'run', 'qb_sack']
        self.play_end_events = ['tackle', 'touchdown', 'pass_outcome_incomplete','out_of_bounds','qb_sack', 'touchback', 'qb_kneel', 'play_submit','qb_spike',]
        self.run_events = ['handoff']
        self.post_snap_events = [ 'run_pass_option','pass_arrived', 'pass_outcome_caught',  'first_contact', 'dropped_pass', 'play_action', 'run', 'pass_tipped', 'fumble', 'fumble_offense_recovered', 'fumble_defense_recovered','qb_strip_sack', 'lateral']

        self.predict_events = ['pass_forward', 'pass_outcome_incomplete', 'pass_outcome_caught', 'pass_tipped', 'pass_outcome_interception', 'pass_outcome_touchdown']
            
    def get_play_events(self, game_id: int, play_id: int) -> List[PlayEvent]:
        """Get all events for a specific play in chronological order."""
        play_data = self.df_tracking[
            (self.df_tracking['gameId'] == game_id) & 
            (self.df_tracking['playId'] == play_id) &
            (self.df_tracking["displayName"] == "football")
        ]
        if len(play_data) == 0:
            return []
        events = []
        for frame_id, frame_data in play_data.groupby('frameId'):
            if not frame_data['event'].isna().all():
                event_type = frame_data['event'].iloc[0]
                
                # Determine phase
                if event_type in self.snap_events:
                    phase = PlayPhase.POST_SNAP
                elif event_type in self.pass_events:
                    phase = PlayPhase.PASS_DECISION
                elif event_type in self.play_end_events:
                    phase = PlayPhase.PLAY_END
                elif event_type in self.pre_snap_events:
                    phase = PlayPhase.PRE_SNAP
                

                
                events.append(PlayEvent(frame_id, event_type, phase))
        
        return sorted(events, key=lambda x: x.frame_id)
    
    def get_player_positions(self, game_id: int, play_id: int, frame_id: int) -> pd.DataFrame:
        """Get player positions for a specific frame."""
        frame_data = self.df_tracking[
            (self.df_tracking['gameId'] == game_id) & 
            (self.df_tracking['playId'] == play_id) &
            (self.df_tracking['frameId'] == frame_id)
        ]
        
        # Merge with player info
        player_data = pd.merge(
            frame_data,
            self.df_players[['nflId', 'position', 'displayName']],
            on='nflId',
            how='left'
        )
        
        return player_data
    
    def  extract_play_features(self, 
                            game_id: int, 
                            play_id: int,
                            start_phase: PlayPhase = PlayPhase.POST_SNAP,
                            end_phase: PlayPhase = PlayPhase.PLAY_END) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract features for a play between specified phases.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: 
                - DataFrame containing player positions and movements for each frame
                - Dictionary containing play outcomes and metadata
        """
        # Get play events
        events = self.get_play_events(game_id, play_id)
        if len(events) == 0:
            raise ValueError(f"Not in Tracking data")
        
        # Find start and end frames
        start_frame = next((e.frame_id for e in events if e.phase == start_phase), None)
        end_frame = next((e.frame_id for e in events if e.phase == end_phase), None)
        if end_frame is None:
            end_frame = next((e.frame_id for e in events if e.phase == PlayPhase.PLAY_END), None)

        if not start_frame or not end_frame:
            raise ValueError(f"Could not find {start_phase} or {end_phase} phase for play")
        
        # Get play info
        play_info = self.df_plays[
            (self.df_plays['gameId'] == game_id) & 
            (self.df_plays['playId'] == play_id)
        ].iloc[0]
        
        # Get targeted receiver from player_plays
        targeted_receiver = self.df_player_plays[
            (self.df_player_plays['gameId'] == game_id) & 
            (self.df_player_plays['playId'] == play_id) &
            (self.df_player_plays['wasTargettedReceiver'] == True)
        ]['nflId'].iloc[0] if len(self.df_player_plays[
            (self.df_player_plays['gameId'] == game_id) & 
            (self.df_player_plays['playId'] == play_id) &
            (self.df_player_plays['wasTargettedReceiver'] == True)
        ]) > 0 else None
        
        # Extract features for each frame
        frames_data = []
        for frame_id in range(start_frame, end_frame + 1):
            frame_data = self.get_player_positions(game_id, play_id, frame_id)
            frames_data.append(frame_data)
        
        # Combine all frames
        play_data = pd.concat(frames_data)
        yardline_side = play_info["yardlineSide"]
        pos_team = play_info["possessionTeam"]
        yardline_number = play_info["yardlineNumber"]


                # Adjust x to be relative to line of scrimmage
        if yardline_side == pos_team:
            los_x = 100 - yardline_number  # Already in correct format
            # First down is always in the direction of the endzone
        else:
            los_x = yardline_number  # Convert from opponent's perspective
            # First down is always in the direction of the endzone
        play_data['x_relative'] = play_data['x'] + los_x - 110

        # Optionally include LOS and yards to endzone as context
        play_data['line_of_scrimmage'] = los_x
        play_data['yards_to_endzone'] = 100 - play_info['yardlineNumber']  # assuming offense always drives right

        
        # Create metadata dictionary
        metadata = {
            'game_id': game_id,
            'play_id': play_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'play_type': play_info['passResult'],
            'down': play_info['down'],
            'yards_to_go': play_info['yardsToGo'],
            'line_of_scrimmage': los_x,
            'yards_gained': play_info['yardsGained'],
            'pass_result': play_info.get('passResult', None),
            'targeted_receiver_nflId': targeted_receiver,
            'time_to_throw': end_frame - start_frame if 'pass_forward' in [e.event_type for e in events] else None
        }
        
        # Pivot: one row per frame, all players as columns
        players_by_frame = []
        for frame_id, frame_df in play_data.groupby("frameId"):
            offense_roles = ["QB", "C", "G", "T", "FB", "RB", "TE", "WR"]
            defense_roles = ["NT", "DT", "DE", "OLB", "MLB", "ILB", "LB", "CB", "DB", "SS", "FS"]
            pos_rank = {pos: i for i, pos in enumerate(offense_roles + defense_roles)}
            frame_df["pos_rank"] = frame_df["position"].map(pos_rank).fillna(999)

            frame_flat = frame_df.sort_values(
                by=["pos_rank", "nflId"], ascending=[True, True]).reset_index(drop=True)

            # Sorting helper

            
            # Flatten player features
            flat_features = {}
            for i, row in frame_flat.iterrows():
                
                prefix = f"player_{i}"
                for col in ["nflId", "displayName_x","position", "x", "x_relative", "y", "s", "a", "dis", "o", "dir"]:
                    flat_features[f"{prefix}_{col}"] = row[col]
            
            # Add metadata for the frame
            flat_features.update({
                "gameId": game_id,
                "playId": play_id,
                "frameId": frame_id,
                "yards_to_go": play_info['yardsToGo'],
                "down": play_info['down'],
                "los": los_x
            })
            players_by_frame.append(flat_features)

        flat_df = pd.DataFrame(players_by_frame)

        return flat_df, metadata
    
    def create_training_dataset(self,
                              start_phase: PlayPhase = PlayPhase.POST_SNAP,
                              end_phase: PlayPhase = PlayPhase.PASS_DECISION) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a dataset for training a model.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - X: Features (player positions and movements)
                - y: Targets (pass decision, completion, yards gained, etc.)
        """
        X_data = []
        y_data = []
        
        
        # Get all passing plays
        passing_plays = self.df_plays[self.df_plays['isDropback'] == True]
        # passing_plays = passing_plays.head(10)
        week_games = self.df_tracking["gameId"].unique()
        for _, play in passing_plays.iterrows():
            try:
                # Extract features for this play
                game_id = play['gameId']
                if game_id not in week_games: #fixme 
                    continue
                play_features, metadata = self.extract_play_features(
                    game_id,
                    play['playId'],
                    start_phase,
                    end_phase
                )
                # print(play_features.shape)
                # Add to dataset
                X_data.append(play_features)
                y_data.append(metadata)
                
            except ValueError as e:
                print(f"Skipping play {game_id}, {play['playId']}: {str(e)}")
                continue
        
        # Combine all plays
        X = pd.concat(X_data, ignore_index=True)
        y = pd.DataFrame(y_data)
        return X, y

def main():
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize output files with headers
    # X_headers = pd.DataFrame(columns=['gameId', 'playId', 'frameId', 'nflId', 'displayName', 'position','club', 'x', 'y', 's', 'a', 'dis', 'o', 'dir'])
    X_headers = pd.DataFrame()
    # columns=['game_id', 'play_id', 'start_frame', 'end_frame', 'play_type', 'down', 'yards_to_go', 'yards_gained', 'pass_result', 'targeted_receiver_nflId', 'time_to_throw']
    y_headers = pd.DataFrame()
    



    X_headers.to_csv('data/play_features.csv', index=False)
    y_headers.to_csv('data/play_outcomes.csv', index=False)
    
    # Process each week
    for week in range(1, 10): #fixme extend to 10
        print(f"Processing week {week}...")
        
        # Load data for current week
        df_tracking = pd.read_csv(f'data/tracking_week_{week}.csv')
        df_players = pd.read_csv('data/players.csv')
        df_plays = pd.read_csv('data/plays.csv')
        df_games = pd.read_csv('data/games.csv')
        df_player_plays = pd.read_csv('data/player_play.csv')

        FIELD_LENGTH = 120.0
        FIELD_WIDTH = 53.3

        left_mask = df_tracking['playDirection'] == 'left'
        df_tracking.loc[left_mask, 'x'] = FIELD_LENGTH - df_tracking.loc[left_mask, 'x']
        df_tracking.loc[left_mask, 'y'] = FIELD_WIDTH - df_tracking.loc[left_mask, 'y']
        df_tracking.loc[left_mask, 'o'] = (180 + df_tracking.loc[left_mask, 'o']) % 360
        df_tracking.loc[left_mask, 'dir'] = (180 + df_tracking.loc[left_mask, 'dir']) % 360
        df_tracking['playDirection'] = 'right'
        
        # Create analyzer
        analyzer = PlayAnalyzer(df_tracking, df_players, df_plays, df_games, df_player_plays)
        
        # Create training dataset for current week
        print(f"Creating training dataset for week {week}...")
        X, y = analyzer.create_training_dataset()
        print(X.columns)
        
        # Append to output files
        print(f"Appending week {week} data to output files...")
        X.to_csv('data/play_features.csv', mode='a', header=True, index=False)
        y.to_csv('data/play_outcomes.csv', mode='a', header=True, index=False)
        
        # Clear memory
        del df_tracking, df_players, df_plays, df_games, df_player_plays, analyzer, X, y
        import gc
        gc.collect()
        
        print(f"Completed week {week}")
        
    
    print("All weeks processed successfully!")

if __name__ == "__main__":
    main() 