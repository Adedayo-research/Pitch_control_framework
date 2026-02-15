#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import scipy.signal as signal


def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay',
                           window=7, polyorder=1, maxspeed=12):
    """ calc_player_velocities(team)

    Calculate player velocities in x & y direction, and total player speed at each timestamp of the tracking data

    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter_: type of filter to use when smoothing the velocities. Default is Savitzky-Golay
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1
        maxspeed: max realistic speed (m/s). Outliers above this are set to NaN.

    Returns
    -----------
        team: tracking DataFrame with vx, vy, speed columns added
    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)

    # Get the player ids (e.g., "Home_11", "Away_7")
    player_ids = np.unique([
        c[:-2] for c in team.columns
        if c[:4] in ['Home', 'Away'] and c.endswith(('_x', '_y'))
    ])

    # Calculate timestep between frames (normally 0.04s)
    dt = team['Time [s]'].diff()

    # Find first frame in second half (if it exists)
    if "Period" in team.columns:
        idx2 = team.index[team["Period"] == 2]
        second_half_idx = idx2[0] if len(idx2) > 0 else None
    else:
        second_half_idx = None

    def _savgol_safe(series, win, poly):
        """Apply Savitzky-Golay safely: remove inf, fill NaNs, ensure window/poly valid."""
        s = series.copy()

        # replace infs, then interpolate and fill
        s = (
            s.replace([np.inf, -np.inf], np.nan)
             .interpolate(limit_direction="both")
             .bfill()
             .ffill()
        )

        n = len(s)
        w = int(win)

        # window must be odd and >=3
        if w < 3:
            return s.values
        if w % 2 == 0:
            w += 1

        # window must be <= n
        if w > n:
            w = n if (n % 2 == 1) else n - 1
        if w < 3:
            return s.values

        # polyorder must be < window
        p = int(poly)
        if p >= w:
            p = max(1, w - 1)

        return signal.savgol_filter(s.values, window_length=w, polyorder=p)

    # estimate velocities for players in team
    for player in player_ids:
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        # remove outliers (position glitches)
        if maxspeed and maxspeed > 0:
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed > maxspeed] = np.nan
            vy[raw_speed > maxspeed] = np.nan

        # clean inf/NaN before smoothing
        vx = vx.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
        vy = vy.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")

        # smoothing
        if smoothing:
            if filter_ == 'Savitzky-Golay':
                if second_half_idx is None:
                    vx[:] = _savgol_safe(vx, window, polyorder)
                    vy[:] = _savgol_safe(vy, window, polyorder)
                else:
                    mask1 = vx.index <= second_half_idx
                    mask2 = vx.index >= second_half_idx

                    vx.loc[mask1] = _savgol_safe(vx.loc[mask1], window, polyorder)
                    vy.loc[mask1] = _savgol_safe(vy.loc[mask1], window, polyorder)

                    vx.loc[mask2] = _savgol_safe(vx.loc[mask2], window, polyorder)
                    vy.loc[mask2] = _savgol_safe(vy.loc[mask2], window, polyorder)

            elif filter_ == 'moving average':
                ma_window = np.ones(int(window)) / float(window)

                if second_half_idx is None:
                    vx_vals = vx.ffill().bfill().values
                    vy_vals = vy.ffill().bfill().values
                    vx[:] = np.convolve(vx_vals, ma_window, mode='same')
                    vy[:] = np.convolve(vy_vals, ma_window, mode='same')
                else:
                    mask1 = vx.index <= second_half_idx
                    mask2 = vx.index >= second_half_idx

                    vx1 = vx.loc[mask1].ffill().bfill().values
                    vy1 = vy.loc[mask1].ffill().bfill().values
                    vx2 = vx.loc[mask2].ffill().bfill().values
                    vy2 = vy.loc[mask2].ffill().bfill().values

                    vx.loc[mask1] = np.convolve(vx1, ma_window, mode='same')
                    vy.loc[mask1] = np.convolve(vy1, ma_window, mode='same')
                    vx.loc[mask2] = np.convolve(vx2, ma_window, mode='same')
                    vy.loc[mask2] = np.convolve(vy2, ma_window, mode='same')

        # put results back in the dataframe
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt(vx**2 + vy**2)

    return team


def remove_player_velocities(team):
    """Remove any existing velocity/acceleration columns from the dataframe."""
    cols = [
        c for c in team.columns
        if c.split('_')[-1] in ['vx', 'vy', 'ax', 'ay', 'speed', 'acceleration']
    ]
    if len(cols) > 0:
        team = team.drop(columns=cols)
    return team
