import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_Viz as mviz


def main():
    # -------------------------
    # PATHS
    # -------------------------
    DATADIR = r"C:\Users\25817442.EDGEHILL.000\OneDrive - Edge Hill University\Desktop\Pitch_control\data"
    game_id = 2
    

    # -------------------------
    # ANIMATION SETTINGS
    # -------------------------
    fps_out = 25
    frame_step = 1
    n_grid_cells_x = 25
    field_dimen = (106.0, 68.0)

    # choose which team is "attacking" for the PPCF heatmap:
    # For a full match, this is not perfectly defined every frame. We'll do a simple default:
    attacking_team = "Home"   # change to "Away" if you want the other side

    out_path = os.path.join(DATADIR, f"pitch_control_game{game_id}_step{frame_step}_grid{n_grid_cells_x}.mp4")

    # -------------------------
    # LOAD DATA
    # -------------------------
    tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
    tracking_away = mio.tracking_data(DATADIR, game_id, "Away")
    events = mio.read_event_data(DATADIR, game_id)

     # Convert to metric coordinates (meters) and set single playing direction
    tracking_home = mio.to_metric_coordinates(tracking_home, field_dimen=field_dimen)
    tracking_away = mio.to_metric_coordinates(tracking_away, field_dimen=field_dimen)
    events = mio.to_metric_coordinates(events, field_dimen=field_dimen)

    tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

    # -------------------------
    # VELOCITIES (IMPORTANT)
    # -------------------------
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    # -------------------------
    # GOALKEEPERS + MODEL PARAMS
    # -------------------------
    GK_numbers = (mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away))
    params = mpc.default_model_params()

    # frames to animate (downsample)
    frames = tracking_home.index[::frame_step]
    # ensure both have same frames
    frames = [f for f in frames if f in tracking_away.index]

    # -------------------------
    # INIT FIGURE + ARTISTS
    # -------------------------
    artists = mviz.init_pitchcontrol_animation_artists(
        field_dimen=field_dimen,
        cmap="bwr",
        heatmap_alpha=0.5,
        team_colors=("r", "b"),
        player_markersize=10,
        player_alpha=0.7
    )
    fig = artists["fig"]

    # Because our init created a placeholder heatmap, we should set its shape to match our PPCF grid once.
    # Create one PPCF to size the heatmap correctly (first frame).
    first_frame = frames[0]
    PPCF0, xgrid, ygrid = mpc.generate_pitch_control_for_frame(
        first_frame, tracking_home, tracking_away,
        attacking_team=attacking_team,
        params=params,
        GK_numbers=GK_numbers,
        n_grid_cells_x=n_grid_cells_x,
        field_dimen=field_dimen,
        offsides=False
    )
    artists["heat"].set_data(PPCF0)

    # -------------------------
    # UPDATE FUNCTION
    # -------------------------
    def update(k):
        frame = frames[k]
        home_row = tracking_home.loc[frame]
        away_row = tracking_away.loc[frame]

        # Compute pitch control for this frame
        PPCF, _, _ = mpc.generate_pitch_control_for_frame(
            frame, tracking_home, tracking_away,
            attacking_team=attacking_team,
            params=params,
            GK_numbers=GK_numbers,
            n_grid_cells_x=n_grid_cells_x,
            field_dimen=field_dimen,
            offsides=False
        )

        # Update artists (players/ball/heat/time)
        return mviz.update_pitchcontrol_animation_artists(
            artists, home_row, away_row, PPCF, field_dimen=field_dimen
        )

    # -------------------------
    # ANIMATE + SAVE
    # -------------------------
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps_out,
        blit=True
    )

    # Save MP4
    writer = animation.FFMpegWriter(fps=fps_out, metadata={"artist": "Adedayo"}, bitrate=2000)
    ani.save(out_path, writer=writer)

    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
print("------pipeline complete______")