import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional
from matplotlib.patches import Ellipse
from matplotlib.patheffects import withStroke
import matplotlib.patches as patches

def create_football_field() -> tuple:
    """Create a football field plot."""
    fig, ax = plt.subplots(figsize=(12, 6.33))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('Yard Line')
    ax.set_ylabel('Field Width')
    
    # Set field color
    ax.set_facecolor('forestgreen')
    
    # Add yard lines
    for x in range(10, 100, 10):
        ax.axvline(x=x, color='white', linestyle='-', linewidth=2)
        ax.text(x, 2, str(x), color='white', ha='center', fontweight='bold', fontsize=8)
    
    # Add hash marks
    for x in range(10, 100, 1):
        if x % 5 != 0:  # Skip yard lines
            ax.plot([x, x], [0, 1], 'w-', linewidth=1)  # Bottom hash
            ax.plot([x, x], [52.3, 53.3], 'w-', linewidth=1)  # Top hash
    
    # Add end zones
    ax.axvspan(0, 10, color='darkblue', alpha=0.2)
    ax.axvspan(90, 100, color='darkblue', alpha=0.2)
    
    return fig, ax

def load_data_chunked(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """Load data in chunks to manage memory usage."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # First check for NaN values in nflId
        if chunk['nflId'].isna().any():
            # Fill NaN values with a placeholder (-1)
            chunk['nflId'] = chunk['nflId'].fillna(-1)
        
        # Process each chunk with error handling
        try:
            chunk = chunk.astype({
                'playId': 'int32',
                'nflId': 'int32',
                'frameId': 'int16',
                'x': 'float32',
                'y': 'float32'
            })
        except (ValueError, TypeError) as e:
            print(f"Warning: Error during type conversion: {e}")
            print("Skipping type conversion for problematic columns")
            # Continue with original types if conversion fails
            pass
            
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def get_play_data(game_id: int, play_id: int,
                 df_tracking: pd.DataFrame,
                 df_players: pd.DataFrame,
                 df_plays: pd.DataFrame,
                 df_games: pd.DataFrame) -> pd.DataFrame:
    """Get data for a specific play efficiently."""
    # Get tracking data for this play
    play_tracking = df_tracking[
        (df_tracking['gameId'] == game_id) & 
        (df_tracking['playId'] == play_id)
    ].copy()
    
    # Get player info
    play_players = df_players.copy()
    
    # Get play info
    play_info = df_plays[
        (df_plays['gameId'] == game_id) & 
        (df_plays['playId'] == play_id)
    ].copy()
    
    # Get game info
    game_info = df_games[df_games['gameId'] == game_id].copy()
    
    # Merge data
    play_data = pd.merge(play_tracking, play_players, on='nflId', how='left')
    play_data = pd.merge(play_data, play_info, on=['gameId', 'playId'], how='left')
    play_data = pd.merge(play_data, game_info, on='gameId', how='left')
    
    return play_data

def get_play_phases(play_data: pd.DataFrame) -> Dict[str, int]:
    """Identify key phases of the play."""
    phases = {}
    
    # Find snap frame
    snap_frames = play_data[play_data['event'] == 'ball_snap']['frameId']
    if not snap_frames.empty:
        phases['snap'] = snap_frames.iloc[0]
    
    # Find end of play frame (tackle, touchdown, etc.)
    end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble', 'pass_outcome_incomplete']
    end_frames = play_data[play_data['event'].isin(end_events)]['frameId']
    if not end_frames.empty:
        phases['end'] = end_frames.iloc[0]
    
    return phases

def animate_play(game_id: int, play_id: int,
                df_tracking: pd.DataFrame,
                df_players: pd.DataFrame,
                df_plays: pd.DataFrame,
                df_games: pd.DataFrame,
                show_labels: str = 'number',  # 'number' or 'position'
                save_path: Optional[str] = None) -> None:
    """Animate player tracking data for a specific play."""
    # Get data for this play
    play_data = get_play_data(game_id, play_id, df_tracking, df_players, df_plays, df_games)
    
    # Get play phases
    phases = get_play_phases(play_data)
    
    # Create football field
    fig, ax = create_football_field()
    
    # Initialize player positions and labels
    players = ax.scatter([], [], s=100)
    player_labels = []
    
    # Initialize football as a scatter point
    football = ax.scatter([], [], s=50, color='#8B4513', zorder=10)  # Saddle brown color
    
    # Initialize lines (TV-style colors)
    los = ax.axvline(x=0, color='#00A6FF', linestyle='-', alpha=0.7, linewidth=3)  # TV blue
    first_down = ax.axvline(x=0, color='#FFD700', linestyle='-', alpha=0.7, linewidth=3)  # TV yellow
    next_play = ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=2)
    
    # Get play information
    try:
        play_info = df_plays[
            (df_plays['gameId'] == game_id) & 
            (df_plays['playId'] == play_id)
        ].iloc[0]
        
        play_description = play_info['playDescription']
        yards_gained = play_info['yardsGained']
        down = play_info['down']
        yards_to_go = play_info['yardsToGo']
        
        # Get team info
        home_team = play_info['possessionTeam']
        away_team = play_info['defensiveTeam']
        
        # Calculate line of scrimmage and first down marker
        yardline_side = play_info['yardlineSide']
        yardline_number = play_info['yardlineNumber']
        
        # Convert to absolute yard line (0-100 scale)
        # The yardline_number is how many yards from the endzone
        if yardline_side == home_team:
            los_x = yardline_number  # Already in correct format
            # First down is always in the direction of the endzone
            first_down_x = los_x + yards_to_go if yardline_number < 50 else los_x - yards_to_go
        else:
            los_x = 100 - yardline_number  # Convert from opponent's perspective
            # First down is always in the direction of the endzone
            first_down_x = los_x - yards_to_go if yardline_number < 50 else los_x + yards_to_go
            
        # Calculate next play line (if yards were gained)
        if yards_gained > 0:
            if yardline_side == home_team:
                next_play_x = los_x + yards_gained
            else:
                next_play_x = los_x - yards_gained
        else:
            next_play_x = los_x  # No gain or incomplete
            
        # Ensure lines are within field bounds
        los_x = max(0, min(100, los_x))
        first_down_x = max(0, min(100, first_down_x))
        next_play_x = max(0, min(100, next_play_x))
        
    except (KeyError, IndexError):
        los_x = 40  # Default to 40 yard line
        first_down_x = los_x + 10
        next_play_x = los_x
        play_description = "Play information not available"
        yards_gained = 0
        down = 1
        yards_to_go = 10
        home_team = 'HOME'
        away_team = 'AWAY'
    
    los.set_xdata([los_x, los_x])
    first_down.set_xdata([first_down_x, first_down_x])
    next_play.set_xdata([next_play_x, next_play_x])
    
    # Create title text
    title_text = f"Game {game_id}, Play {play_id}\n"
    title_text += f"{play_description}\n"
    title_text += f"Down: {down}, Distance: {yards_to_go} yards, Gain: {yards_gained} yards"
    print("Title: ", title_text)
    
    # Create a text object for the title
    title = ax.text(0.5, 1.05, title_text, 
                   transform=ax.transAxes,
                   ha='center', va='bottom',
                   color='white', fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    # Create event text
    event_text = ax.text(0.5, 0.95, "", 
                        transform=ax.transAxes,
                        ha='center', va='top',
                        color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    def update(frame: int) -> List:
        """Update player positions for each frame."""
        frame_data = play_data[play_data['frameId'] == frame]
        
        # Separate players and football
        players_data = frame_data[
            (frame_data['nflId'].notna()) & 
            (frame_data['club'].notna()) & 
            (frame_data['club'] != 'football')
        ]
        
        # Football has specific characteristics
        football_data = frame_data[
            (frame_data['nflId'].isna()) | 
            (frame_data['club'] == 'football') | 
            (frame_data['jerseyNumber'].isna())
        ]
        
        # Update scatter plot for players
        players.set_offsets(np.c_[players_data['x'] - 10, players_data['y']])
        
        # Set team colors
        colors = ['red' if team == home_team else 'blue' 
                 for team in players_data['club']]
        players.set_color(colors)
        
        # Update player labels
        for label in player_labels:
            label.remove()
        player_labels.clear()
        
        for idx, row in players_data.iterrows():
            if show_labels == 'number':
                try:
                    jersey_num = int(row['jerseyNumber'])
                    label_text = str(jersey_num)
                except (ValueError, TypeError):
                    label_text = '?'
            else:  # position
                label_text = row['position']
            
            label = ax.text(row['x'] - 10, row['y'], label_text,
                          color='white', ha='center', va='center',
                          fontsize=6, fontweight='bold',
                          path_effects=[withStroke(linewidth=2, foreground='black')])
            player_labels.append(label)
        
        # Update football position
        if not football_data.empty:
            football.set_offsets(np.c_[football_data['x'].iloc[0] - 10, football_data['y'].iloc[0]])
            football.set_visible(True)
        else:
            football.set_visible(False)
        
        # Update title with play phase
        phase = "Pre-snap"
        if 'snap' in phases and frame >= phases['snap']:
            phase = "Post-snap"
        if 'end' in phases and frame >= phases['end']:
            phase = "Play Complete"
        
        title.set_text(f"{title_text}\nPhase: {phase}")
        
        # Update event text
        current_event = frame_data['event'].iloc[0] if not frame_data['event'].isna().all() else ""
        if current_event:
            event_text.set_text(f"Event: {current_event}")
        else:
            event_text.set_text("")
        
        return [players, football, los, first_down, next_play, title, event_text] + player_labels
    
    # Create animation
    ani = animation.FuncAnimation(fig, 
                                update,
                                frames=play_data['frameId'].unique(),
                                interval=100,
                                blit=True)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    else:
        plt.show()

def main():
    # Load data in chunks
    print("Loading tracking data...")
    df_tracking = load_data_chunked('data/tracking_week_1.csv')
    
    print("Loading player data...")
    df_players = pd.read_csv('data/players.csv')
    
    print("Loading play data...")
    df_plays = pd.read_csv('data/plays.csv')
    
    print("Loading game data...")
    df_games = pd.read_csv('data/games.csv')
    
    # Example game and play IDs
    game_id = 2022091200
    play_id = 180
    #  64,   85,  109,  156,  180,  201,  264,  286,  315,  346,  375,
    #     401,  446,  467,  565,  601,  622,  643,  664,  688,  716,  741,
    #     762,  786,  810,  882,  910,  931,  983, 1004, 1028, 1057, 1092,
    #    1164, 1217, 1241, 1299, 1320, 1344, 1409, 1433, 1465, 1487, 1521,
    #    1550, 1579, 1642, 1680, 1704, 1725, 1764, 1793, 1815, 1851, 1967,
    #    1988, 2009, 2038, 2067, 2093, 2188, 2244, 2268, 2292, 2370, 2391,
    #    2479, 2500, 2522, 2546, 2591, 2613, 2667, 2688, 2712, 2750, 2779,
    #    2801, 2830, 2883, 2923, 2944, 2965, 3001, 3048, 3077, 3101, 3125,
    #    3149, 3173, 3194, 3216, 3245, 3267, 3296, 3325, 3382, 3404, 3467,
    #    3491, 3515, 3553, 3574, 3596, 3628, 3685, 3723, 3747, 3795, 3826,
    #    3980, 4012
    
    print(f"Animating game {game_id}, play {play_id}...")
    animate_play(game_id, play_id, df_tracking, df_players, df_plays, df_games, show_labels='position')

if __name__ == "__main__":
    main() 