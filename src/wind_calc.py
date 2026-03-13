"""Apparent Wind Speed (AWS) and Apparent Wind Angle (AWA) vector calculation."""

import numpy as np


def compute_apparent_wind(df):
    """
    Calculate AWS and AWA from True Wind (TWS/TWD) and boat motion (SOG/COG).

    Convention:
    - TWD: direction wind is coming FROM (meteorological, 0-360)
    - COG: direction boat is heading (0-360)
    - AWA: angle relative to bow, negative=port, positive=starboard (-180 to +180)
    """
    twd_rad = np.radians(df['TWD'].values)
    cog_rad = np.radians(df['COG'].values)
    tws = df['TWS'].values
    sog = df['SOG'].values

    # True wind velocity vector (direction wind is blowing TO)
    tx = -tws * np.sin(twd_rad)
    ty = -tws * np.cos(twd_rad)

    # Boat velocity vector
    bx = sog * np.sin(cog_rad)
    by = sog * np.cos(cog_rad)

    # Apparent wind = true wind - boat motion (relative to boat)
    ax = tx - bx
    ay = ty - by

    aws = np.sqrt(ax**2 + ay**2)

    # AWA relative to boat heading
    # Rotate apparent vector into boat frame
    awa_abs = np.degrees(np.arctan2(ax, ay))
    # Convert to relative angle (-180 to +180 from boat heading)
    awa = awa_abs - df['COG'].values
    # Normalize to -180..+180
    awa = (awa + 180) % 360 - 180

    df['AWS'] = aws
    df['AWA'] = awa
    return df


def run(config, df):
    """Compute apparent wind."""
    print("[Step 3] Apparent wind calculation...")
    df = compute_apparent_wind(df)
    print(f"  AWS range: {df['AWS'].min():.1f} ~ {df['AWS'].max():.1f} kn")
    print(f"  AWA range: {df['AWA'].min():.1f} ~ {df['AWA'].max():.1f}°")
    return df
