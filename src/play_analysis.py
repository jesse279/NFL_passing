import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

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
                 df_games: pd.DataFrame):
        self.df_tracking = df_tracking
        self.df_players = df_players
        self.df_plays = df_plays
        self.df_games = df_games
        
        # Define key events for different phases
        self.snap_events = ['ball_snap']
        self.pass_events = ['pass_forward', 'pass_arrived', 'pass_outcome_incomplete', 'pass_outcome_complete']
        self.play_end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble', 'pass_outcome_incomplete', 'pass_outcome_complete']
    
    def get_play_events(self, game_id: int, play_id: int) -> List[PlayEvent]:
        """Get all events for a specific play in chronological order."""
        play_data = self.df_tracking[
            (self.df_tracking['gameId'] == game_id) & 
            (self.df_tracking['playId'] == play_id)
        ]
        
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
                else:
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
    
    def extract_play_features(self, 
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
        
        # Find start and end frames
        start_frame = next((e.frame_id for e in events if e.phase == start_phase), None)
        end_frame = next((e.frame_id for e in events if e.phase == end_phase), None)
        
        if not start_frame or not end_frame:
            raise ValueError(f"Could not find {start_phase} or {end_phase} phase for play")
        
        # Get play info
        play_info = self.df_plays[
            (self.df_plays['gameId'] == game_id) & 
            (self.df_plays['playId'] == play_id)
        ].iloc[0]
        
        # Extract features for each frame
        frames_data = []
        for frame_id in range(start_frame, end_frame + 1):
            frame_data = self.get_player_positions(game_id, play_id, frame_id)
            frames_data.append(frame_data)
        
        # Combine all frames
        play_data = pd.concat(frames_data)
        
        # Create metadata dictionary
        metadata = {
            'game_id': game_id,
            'play_id': play_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'play_type': play_info['playType'],
            'down': play_info['down'],
            'yards_to_go': play_info['yardsToGo'],
            'yards_gained': play_info['yardsGained'],
            'pass_result': play_info.get('passResult', None),
            'targeted_receiver': play_info.get('targetedReceiver', None),
            'time_to_throw': end_frame - start_frame if 'pass_forward' in [e.event_type for e in events] else None
        }
        
        return play_data, metadata
    
    def create_training_dataset(self,
                              start_phase: PlayPhase = PlayPhase.POST_SNAP,
                              end_phase: PlayPhase = PlayPhase.PLAY_END) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        
        for _, play in passing_plays.iterrows():
            try:
                # Extract features for this play
                play_features, metadata = self.extract_play_features(
                    play['gameId'],
                    play['playId'],
                    start_phase,
                    end_phase
                )
                
                # Add to dataset
                X_data.append(play_features)
                y_data.append(metadata)
                
            except ValueError as e:
                print(f"Skipping play {play['playId']}: {str(e)}")
                continue
        
        # Combine all plays
        X = pd.concat(X_data, ignore_index=True)
        y = pd.DataFrame(y_data)
        
        return X, y

def main():
    # Load data
    print("Loading data...")
    df_tracking = pd.read_csv('data/tracking_week_1.csv')
    df_players = pd.read_csv('data/players.csv')
    df_plays = pd.read_csv('data/plays.csv')
    df_games = pd.read_csv('data/games.csv')
    
    # Create analyzer
    analyzer = PlayAnalyzer(df_tracking, df_players, df_plays, df_games)
    
    # Create training dataset
    print("Creating training dataset...")
    X, y = analyzer.create_training_dataset()
    
    # Save datasets
    print("Saving datasets...")
    X.to_csv('data/play_features.csv', index=False)
    y.to_csv('data/play_outcomes.csv', index=False)
    
    print("Done!")

if __name__ == "__main__":
    main() 