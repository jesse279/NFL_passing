import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional
from matplotlib.patches import Ellipse
from matplotlib.patheffects import withStroke
import matplotlib.patches as patches

import torch

import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 230),
            nn.ReLU(),
            nn.BatchNorm1d(230),

            nn.Dropout(0.4),
            nn.Linear(230, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

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

    # Field dimensions
    FIELD_LENGTH = 120.0
    FIELD_WIDTH = 53.3



    # Mask for plays going left
    left_mask = play_tracking['playDirection'] == 'left'

    # Flip x and y positions
    play_tracking.loc[left_mask, 'x'] = FIELD_LENGTH - play_tracking.loc[left_mask, 'x']
    play_tracking.loc[left_mask, 'y'] = FIELD_WIDTH - play_tracking.loc[left_mask, 'y']

    # Flip orientation and direction: (angle mirror)
    play_tracking.loc[left_mask, 'o'] = (180 + play_tracking.loc[left_mask, 'o']) % 360
    play_tracking.loc[left_mask, 'dir'] = (180 + play_tracking.loc[left_mask, 'dir']) % 360

    # After flipping, you can optionally set all directions to "right"
    play_tracking['playDirection'] = 'right'
    
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

def animate_play(
    game_id: int,
    play_id: int,
    df_tracking: pd.DataFrame,
    df_players: pd.DataFrame,
    df_plays: pd.DataFrame,
    df_games: pd.DataFrame,
    show_labels: str = "number",  # default fallback
    prediction_mode: Optional[str] = None,  # "completion", "interception", "exp_yards_completed", "exp_yards_targeted"
    pred_df: Optional[pd.DataFrame] = None,  # precomputed model predictions for this play
    save_path: Optional[str] = None
) -> None:
    """Animate player tracking data for a specific play, optionally overlaying model predictions."""

    # === Get play tracking data
    play_data = get_play_data(game_id, play_id, df_tracking, df_players, df_plays, df_games)
    phases = get_play_phases(play_data)
    fig, ax = create_football_field()

    players = ax.scatter([], [], s=100)
    player_labels = []
    football = ax.scatter([], [], s=50, color='#8B4513', zorder=10)

    los = ax.axvline(x=0, color='#00A6FF', linestyle='-', alpha=0.7, linewidth=3)
    first_down = ax.axvline(x=0, color='#FFD700', linestyle='-', alpha=0.7, linewidth=3)
    next_play = ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=2)

    # === Extract play meta info
    try:
        play_info = df_plays[(df_plays['gameId'] == game_id) & (df_plays['playId'] == play_id)].iloc[0]
        play_description = play_info['playDescription']
        yards_gained = play_info['yardsGained']
        down = play_info['down']
        yards_to_go = play_info['yardsToGo']
        pos_team = play_info['possessionTeam']
        yardline_number = play_info['yardlineNumber']
        yardline_side = play_info["yardlineSide"]
        if yardline_side == pos_team:
            los_x = yardline_number  # Already in correct format
            # First down is always in the direction of the endzone
        else:
            los_x = 100 - yardline_number  # Convert from opponent's perspective
        first_down_x = los_x + yards_to_go
        next_play_x = los_x + yards_gained if yards_gained != 0 else los_x
    except (KeyError, IndexError):
        los_x, first_down_x, next_play_x = 40, 50, 40
        play_description, yards_gained, down, yards_to_go, pos_team = (
            "Play information not available", 0, 1, 10, "HOME"
        )

    los.set_xdata([los_x, los_x])
    first_down.set_xdata([first_down_x, first_down_x])
    next_play.set_xdata([next_play_x, next_play_x])

    # === Title
    title_text = f"Game {game_id}, Play {play_id}\n{play_description}\nDown: {down}, Distance: {yards_to_go} yards"
    title = ax.text(0.5, 1.05, title_text, transform=ax.transAxes, ha='center', va='bottom',
                    color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    event_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha='center', va='top',
                         color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # === Helper to get label from pred_df
    def get_prediction_label(frame_id, player_id):
        if pred_df is None or prediction_mode is None:
            return None
        row = pred_df[
            (pred_df["frame_id"] == frame_id) &
            (pred_df["receiver_id"] == player_id)
        ]
        if row.empty:
            return None

        r = row.iloc[0]
        if prediction_mode == "completion":
            return f"{100 * r['prob_complete']:.1f}%"
        elif prediction_mode == "interception":
            return f"INT {100 * r['prob_interception']:.1f}%"
        elif prediction_mode == "exp_yards_completed":
            return f"{r['exp_yards_if_completed']:.1f}y"
        elif prediction_mode == "exp_yards_targeted":
            # Expected yards *if targeted* = prob_complete * yards_if_completed
            return f"{r['prob_complete'] * r['exp_yards_if_completed']:.1f}y"
        return None

    def update(frame: int) -> List:
        frame_data = play_data[play_data['frameId'] == frame]

        # === Player scatter positions
        players_data = frame_data[(frame_data['nflId'].notna()) & (frame_data['club'] != 'football')]
        football_data = frame_data[(frame_data['club'] == 'football') | (frame_data['jerseyNumber'].isna())]

        players.set_offsets(np.c_[players_data['x'] - 10, players_data['y']])
        colors = ['red' if team == pos_team else 'blue' for team in players_data['club']]
        players.set_color(colors)

        # === Clear and re-add labels
        for label in player_labels:
            label.remove()
        player_labels.clear()

        for idx, row in players_data.iterrows():
            receiver_id = int(row['nflId'])
            base_label = None

            # Either default jersey/position label OR prediction
            if prediction_mode is None:
                if show_labels == "number":
                    base_label = str(int(row['jerseyNumber'])) if pd.notna(row['jerseyNumber']) else "?"
                else:
                    base_label = row['position']
            else:
                pred_label = get_prediction_label(frame, receiver_id)
                if pred_label:
                    base_label = pred_label
                else:
                    # fallback to jersey
                    base_label = str(int(row['jerseyNumber'])) if pd.notna(row['jerseyNumber']) else "?"

            label = ax.text(
                row['x'] - 10, row['y'], base_label,
                color='white', ha='center', va='center',
                fontsize=6, fontweight='bold',
                path_effects=[withStroke(linewidth=2, foreground='black')]
            )
            player_labels.append(label)

        # Football update
        if not football_data.empty:
            football.set_offsets(np.c_[football_data['x'].iloc[0] - 10, football_data['y'].iloc[0]])
            football.set_visible(True)
        else:
            football.set_visible(False)

        # Title phase
        phase = "Pre-snap"
        if 'snap' in phases and frame >= phases['snap']:
            phase = "Post-snap"
        if 'end' in phases and frame >= phases['end']:
            phase = "Play Complete"
        title.set_text(f"{title_text}\nPhase: {phase}")

        # Event label
        current_event = frame_data['event'].iloc[0] if not frame_data['event'].isna().all() else ""
        event_text.set_text(f"Event: {current_event}" if current_event else "")

        return [players, football, los, first_down, next_play, title, event_text] + player_labels

    # === Animate
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=play_data['frameId'].unique(),
        interval=100,
        blit=True
    )

    if save_path:
        ani.save(f"{save_path}_{game_id}_{play_id}_{prediction_mode}.mp4", writer='ffmpeg', fps=10)


    else:
        plt.show()


def get_model_predictions(game_id=2022110300, play_id=2926):
    # === Load models ===
    clf_model = Net(195, 3)
    reg_model = Net(195, 1)
    clf_model.load_state_dict(torch.load("models/throw_classifier_v_1.pt"))
    reg_model.load_state_dict(torch.load("models/throw_regressor_v_1.pt"))
    clf_model.eval()
    reg_model.eval()

    # === Load scaled export features ===
    export_features = pd.read_csv("data/export_features.csv")

    # Filter for this specific play
    play_df = export_features[
        (export_features["game_id"] == game_id) &
        (export_features["play_id"] == play_id)
    ].copy()


    # Columns not used by the model
    cols_to_drop = [
        col for col in export_features.columns
        if col.endswith("nflId")
        or col in ["game_id", "play_id", "frame_id", "yards_gained", "pass_result"]
    ]

    receiver_range = range(6, 11)  # players 6–10
    receiver_attrs = ['x', 'x_relative', 'y', 's', 'a', 'dis', 'o', 'dir']

    results = []  # will store (frame_id, receiver_id, prob_complete, prob_incomplete, prob_int, exp_yards)

    for frame_id, frame_data in play_df.groupby("frame_id"):
        # Frame-level results
        frame_results = []

        # Base features for this frame (before perturbation)
        base_row = frame_data.iloc[0].copy()  # assumes 1 row/frame (per QB)
        
        # Collect all modified receiver rows for this frame
        modified_rows = []
        receiver_ids = []

        for receiver_idx in receiver_range:
            modified_row = base_row.copy()

            # Replace targeted receiver features with this candidate receiver
            for attr in receiver_attrs:
                modified_row[f"target_receiver_index"] = receiver_idx - 6

                modified_row[f"target_{attr}"] = base_row[f"player_{receiver_idx}_{attr}"]
            modified_row.drop(cols_to_drop, inplace=True)
            modified_row = pd.to_numeric(modified_row, errors='coerce')

            modified_rows.append(modified_row)
            receiver_ids.append(int(base_row[f"player_{receiver_idx}_nflId"]))  # keep receiver_id

        # Convert entire batch of candidate receivers → tensor
        batch_np = np.stack([r.values for r in modified_rows])
        batch_tensor = torch.tensor(batch_np, dtype=torch.float32)

        # === Run through models once for all receivers ===
        with torch.no_grad():
            logits = clf_model(batch_tensor)
            probs = torch.softmax(logits, dim=1).numpy()

            reg_preds = reg_model(batch_tensor).squeeze().numpy()  # expected yards if caught

        # Collect results for this frame
        for rid, p, yhat in zip(receiver_ids, probs, reg_preds):
            results.append({
                "frame_id": frame_id,
                "receiver_id": rid,
                "prob_complete": p[0],
                "prob_incomplete": p[1],
                "prob_interception": p[2],
                "exp_yards_if_completed": yhat
            })

    # Convert to DataFrame for easy saving
    pred_df = pd.DataFrame(results)
    
    

    
    return pred_df





def main():

    
    print("Loading tracking data...")
    df_tracking = load_data_chunked('data/tracking_week_1.csv')
    
    print("Loading player data...")
    df_players = pd.read_csv('data/players.csv')
    
    print("Loading play data...")
    df_plays = pd.read_csv('data/plays.csv')
    
    print("Loading game data...")
    df_games = pd.read_csv('data/games.csv')
    
    # Example game and play IDs


# 85 6 yard pass

# 3574 short incomplete

# 1465 short incomplete pred yards high

# 109 short pass. 

# 286 short pass for big yards. Model predicts high expected yards

# 2093 lines messed up. Check down, high yards predicted, 1 yard gained.

# 3491 pass behind the line of scrimmage, predicted 6 yards got 1.

# 1344 60 yard pass, model drastically increases prediction in last 2 frames. Could be overfitting. 

# 565 short pass to wide open reciever.

# 1320 screen pass for 8. Model prediction jumps as rb clears defense

# 3001 comeback route incomplete. Expected yards if completed is high while it looks like a streak, but drops when the reciever comesback.
# has a high interception rate if blanketed by a reciever. A wideopen rb has 3% interception rate, but there are DL near QB


# 2244 goalline dump pass for 2 yards. 50%

# 3048 dump pass with pressure in face. 2 yards off 70%

# 1579 25 yard tochdown 60% chance

# 2268 texas route dump. 8 yards 48%

# 1764 10 yard pass with extra yards gained. 50% jumps to 17 expected yards once the reciever steps back

# 201 QB extends play passes to reciever behind coverage. predicts 8 yards

# 3296 QB throws dump pass under pressure

# 3101 17 yards 60% sitting in hole of zone

# 1241 6 yard hook 90% competion probability. Defense is soft.

# 1487 17 yards. pred 18. competion percentage is 40, but increase to 70 once he gets a step behind defender




    game_id = 2022091111																			#	2022091200
    play_id = 1840

# array([3574., 1465.,  109.,  286., 2093., 3491., 1344.,  565., 1320.,
#        3001., 2244., 3048., 1579., 2268., 1764.,  201., 3296., 3101.,
#        1241., 1487.,  401., 3382., 1793., 2038., 2688., 2801., 3216.,
#        2009., 3325., 1550., 2500.,   85., 2188.,  156., 1521., 3267.,
#        1028., 2522., 3747.,  786., 1004.,  346.,  664., 2750., 3125.,
#        3149., 3628., 2546., 3723.,  467.,  762., 3826., 2591., 1725.,
#        3245., 2779., 1851.,  688., 1680., 3194.,  931., 3467., 1057.,
#         264., 3596., 1217., 1409., 3404.])

    # -61 yards 2022091811	4519

        # Load data in chunks
    print("Loading Predictions")
    pred_df = get_model_predictions(game_id=game_id, play_id=play_id)

    print(f"Animating game {game_id}, play {play_id}...")
    animate_play(
    game_id=game_id,
    play_id=play_id,
    df_tracking=df_tracking,
    df_players=df_players,
    df_plays=df_plays,
    df_games=df_games,
    prediction_mode="exp_yards_completed",  # can be "completion", "interception", "exp_yards_completed", "exp_yards_targeted"
    pred_df=pred_df,
    show_labels="number",
    save_path="outputs/animations/"
)

if __name__ == "__main__":
    main() 