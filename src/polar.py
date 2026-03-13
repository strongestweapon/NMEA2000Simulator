"""Bavaria 56 polar diagram and wind correction via polar inversion + Kalman filter.

Polar data is an approximation for Bavaria 56 Cruiser (~18t displacement).
Sail modes: Headsail+Main (upwind/reach), A2 Spinnaker (broad reach/run).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d


# --- Bavaria 56 Polar Table ---
# TWA in degrees (true wind angle, 0=head to wind)
POLAR_TWA = np.array([30, 40, 52, 60, 75, 90, 110, 120, 135, 150, 165, 180])

# TWS in knots (extended to 30kn for strong wind inversion)
POLAR_TWS = np.array([4, 6, 8, 10, 12, 14, 16, 20, 25, 30])

# BSP (boat speed in knots) for Headsail + Main
# Shape: (TWA, TWS)
POLAR_MAIN = np.array([
    # TWS:  4    6    8   10   12   14   16   20   25   30
    [2.5, 3.5, 4.5, 5.0, 5.3, 5.5, 5.6, 5.6, 5.5, 5.3],  # 30°
    [3.2, 4.5, 5.5, 6.2, 6.5, 6.8, 7.0, 7.0, 6.8, 6.5],  # 40°
    [3.8, 5.2, 6.2, 6.8, 7.2, 7.4, 7.6, 7.5, 7.3, 7.0],  # 52°
    [4.0, 5.5, 6.5, 7.0, 7.5, 7.7, 7.9, 7.8, 7.5, 7.2],  # 60°
    [4.2, 5.6, 6.6, 7.2, 7.7, 8.0, 8.2, 8.2, 8.0, 7.8],  # 75°
    [4.0, 5.5, 6.5, 7.2, 7.8, 8.2, 8.5, 8.8, 8.5, 8.3],  # 90°
    [3.5, 5.0, 6.0, 6.8, 7.5, 8.0, 8.3, 8.8, 9.0, 9.0],  # 110°
    [3.2, 4.8, 5.8, 6.5, 7.2, 7.8, 8.0, 8.5, 8.8, 8.8],  # 120°
    [2.8, 4.2, 5.3, 6.0, 6.7, 7.3, 7.6, 8.0, 8.2, 8.3],  # 135°
    [2.5, 3.8, 4.8, 5.5, 6.2, 6.8, 7.2, 7.5, 7.8, 7.9],  # 150°
    [2.2, 3.4, 4.3, 5.0, 5.7, 6.2, 6.5, 6.8, 7.0, 7.1],  # 165°
    [2.0, 3.0, 3.8, 4.5, 5.2, 5.7, 6.0, 6.2, 6.3, 6.3],  # 180°
])

# A2 spinnaker bonus (knots added to base polar, effective TWA 80-170°)
SPINNAKER_BONUS = np.array([
    # TWS:  4    6    8   10   12   14   16   20   25   30
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 30°
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 40°
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 52°
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 60°
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 75°
    [0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.0, 0.0],  # 90°
    [0.5, 0.8, 1.0, 1.2, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3],  # 110°
    [0.6, 1.0, 1.3, 1.5, 1.5, 1.4, 1.2, 1.0, 0.7, 0.4],  # 120°
    [0.7, 1.1, 1.4, 1.5, 1.5, 1.4, 1.2, 0.9, 0.6, 0.3],  # 135°
    [0.5, 0.9, 1.2, 1.3, 1.3, 1.2, 1.0, 0.7, 0.4, 0.2],  # 150°
    [0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.0, 0.0],  # 165°
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 180°
])

# Combined polar with spinnaker
POLAR_FULL = POLAR_MAIN + SPINNAKER_BONUS

# Performance factor: real crew vs. theoretical polar (race crew ~90-95%)
PERFORMANCE_FACTOR = 0.92


def _build_interpolator():
    """Build 2D interpolator for BSP = f(TWA, TWS)."""
    return RegularGridInterpolator(
        (POLAR_TWA, POLAR_TWS), POLAR_FULL * PERFORMANCE_FACTOR,
        method='linear', bounds_error=False, fill_value=None
    )


_polar_interp = _build_interpolator()


def lookup_bsp(twa, tws):
    """Look up predicted BSP for given TWA (deg) and TWS (knots).

    Args:
        twa: True wind angle, 0-180° (symmetric)
        tws: True wind speed in knots
    Returns:
        Predicted boat speed in knots
    """
    twa = np.abs(twa)
    twa = np.clip(twa, POLAR_TWA[0], POLAR_TWA[-1])
    tws = np.clip(tws, POLAR_TWS[0], POLAR_TWS[-1])
    pts = np.column_stack([np.atleast_1d(twa), np.atleast_1d(tws)])
    return _polar_interp(pts).flatten()


def invert_tws(twa, sog):
    """Estimate TWS from observed SOG and TWA using polar inversion.

    For a given TWA, find what TWS would produce the observed SOG.
    Uses 1D root finding along the TWS axis.

    Args:
        twa: True wind angle (degrees, 0-180)
        sog: Speed over ground (knots)
    Returns:
        Estimated TWS (knots), or NaN if inversion fails
    """
    twa = np.abs(twa)
    twa_clipped = np.clip(twa, POLAR_TWA[0], POLAR_TWA[-1])

    # Build BSP vs TWS curve for this TWA
    tws_range = np.linspace(POLAR_TWS[0], POLAR_TWS[-1], 100)
    bsp_curve = lookup_bsp(np.full_like(tws_range, twa_clipped), tws_range)

    # Account for performance factor already applied in lookup_bsp
    target_bsp = sog  # SOG ≈ BSP (ignoring current)

    # If SOG is below minimum polar prediction, return minimum TWS
    if target_bsp <= bsp_curve[0]:
        return POLAR_TWS[0]

    # If SOG exceeds maximum polar prediction, return maximum TWS
    if target_bsp >= bsp_curve[-1]:
        return POLAR_TWS[-1]

    # Interpolate: find TWS where BSP = target
    # BSP is generally monotonically increasing with TWS (mostly)
    try:
        inv_func = interp1d(bsp_curve, tws_range, bounds_error=False, fill_value='extrapolate')
        return float(inv_func(target_bsp))
    except Exception:
        return np.nan


def correct_wind_kalman(df, config):
    """Correct ERA5 wind using GPS track + polar inversion + Kalman filter.

    For each point:
    1. Compute TWA from ERA5 TWD and GPS COG
    2. Use polar inversion to estimate TWS from SOG
    3. Kalman filter to fuse ERA5 TWS (prior) with polar-inverted TWS (obs)

    TWD is adjusted proportionally when TWS changes significantly.
    """
    era5_tws = df['TWS'].values.copy()
    era5_twd = df['TWD'].values.copy()
    sog_raw = df['SOG'].values
    cog_raw = df['COG'].values

    # Smooth SOG: rolling median (removes GPS spikes) then rolling mean
    sog_series = pd.Series(sog_raw)
    sog = sog_series.rolling(5, center=True, min_periods=1).median() \
                     .rolling(10, center=True, min_periods=1).mean().values

    # Smooth COG: circular smoothing via sin/cos decomposition
    cog_rad = np.radians(cog_raw)
    sin_smooth = pd.Series(np.sin(cog_rad)).rolling(15, center=True, min_periods=1).mean().values
    cos_smooth = pd.Series(np.cos(cog_rad)).rolling(15, center=True, min_periods=1).mean().values
    cog = np.degrees(np.arctan2(sin_smooth, cos_smooth)) % 360

    n = len(df)
    corrected_tws = np.full(n, np.nan)
    corrected_twd = np.full(n, np.nan)
    gps_implied_tws = np.full(n, np.nan)

    # --- Kalman filter parameters ---
    # Process noise: low = smoother output (real wind changes gradually)
    Q = 0.15 ** 2  # (knots)^2
    # ERA5 measurement noise (high = less trust in ERA5)
    R_era5 = 4.0 ** 2  # ERA5 systematically underestimates (~6kn low)
    # Polar inversion measurement noise
    R_polar = 3.0 ** 2  # Polar + current + GPS residual noise

    # Detect maneuvering: COG variability over a rolling window
    cog_series = pd.Series(cog)
    # Circular std of COG: use sin/cos decomposition
    cog_rad = np.radians(cog)
    sin_mean = pd.Series(np.sin(cog_rad)).rolling(20, center=True, min_periods=5).mean()
    cos_mean = pd.Series(np.cos(cog_rad)).rolling(20, center=True, min_periods=5).mean()
    cog_stability = np.sqrt(sin_mean**2 + cos_mean**2).values  # 1=stable, 0=random
    # Also check SOG stability
    sog_cv = pd.Series(sog).rolling(20, center=True, min_periods=5).std() / \
             pd.Series(sog).rolling(20, center=True, min_periods=5).mean()
    sog_cv = sog_cv.fillna(1.0).values

    # Initialize
    x = era5_tws[0]  # state estimate
    P = R_era5        # state uncertainty

    for i in range(n):
        # --- Prediction step: assume TWS doesn't change much ---
        x_pred = x
        P_pred = P + Q

        # --- TWA from ERA5 TWD + GPS COG ---
        twa = era5_twd[i] - cog[i]
        twa = (twa + 180) % 360 - 180  # normalize to -180..+180

        # --- Polar inversion (only if sailing steadily) ---
        has_polar_obs = False
        is_maneuvering = cog_stability[i] < 0.8 or sog_cv[i] > 0.5
        if not is_maneuvering and sog[i] > 2.0 and abs(twa) > 25:
            implied = invert_tws(twa, sog[i])
            if not np.isnan(implied) and implied > 0:
                gps_implied_tws[i] = implied
                has_polar_obs = True

        # --- Update step: fuse ERA5 + polar inversion ---
        if has_polar_obs:
            # Two observations: ERA5 and polar-inverted
            # Sequential Kalman update

            # Update with ERA5
            K1 = P_pred / (P_pred + R_era5)
            x1 = x_pred + K1 * (era5_tws[i] - x_pred)
            P1 = (1 - K1) * P_pred

            # Update with polar inversion
            K2 = P1 / (P1 + R_polar)
            x = x1 + K2 * (gps_implied_tws[i] - x1)
            P = (1 - K2) * P1
        else:
            # Only ERA5 observation
            K = P_pred / (P_pred + R_era5)
            x = x_pred + K * (era5_tws[i] - x_pred)
            P = (1 - K) * P_pred

        corrected_tws[i] = max(0, x)

        # TWD: use ERA5 directly (sea breeze shifts near coast are real)
        corrected_twd[i] = era5_twd[i]

    # Store results
    df['TWS_era5'] = era5_tws
    df['TWD_era5'] = era5_twd
    df['TWS_gps'] = gps_implied_tws
    df['TWS'] = corrected_tws
    df['TWD'] = corrected_twd

    return df


def run(config, df):
    """Apply polar-based wind correction."""
    print("[Step 2b] Wind correction (Bavaria 56 polar + Kalman filter)...")

    df = correct_wind_kalman(df, config)

    has_gps = ~np.isnan(df['TWS_gps'].values)
    n_corrected = has_gps.sum()
    mean_diff = np.nanmean(df['TWS_gps'].values - df['TWS_era5'].values)
    print(f"  Polar inversion applied to {n_corrected}/{len(df)} points")
    print(f"  Mean TWS difference (GPS-implied - ERA5): {mean_diff:+.1f} kn")
    print(f"  Corrected TWS range: {df['TWS'].min():.1f} ~ {df['TWS'].max():.1f} kn")

    return df
