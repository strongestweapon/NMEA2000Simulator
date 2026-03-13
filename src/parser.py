"""GPX parsing, SOG/COG calculation, outlier filtering, and resampling."""

import math
import gpxpy
import numpy as np
import pandas as pd


def haversine_nm(lat1, lon1, lat2, lon2):
    """Haversine distance in nautical miles."""
    R = 3440.065  # Earth radius in nm
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees (0-360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    brng = math.degrees(math.atan2(x, y))
    return brng % 360


def parse_gpx(gpx_path):
    """Parse GPX file into DataFrame with timestamp, lat, lon, ele."""
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                rows.append({
                    'timestamp_utc': pt.time,
                    'lat': pt.latitude,
                    'lon': pt.longitude,
                    'ele': pt.elevation or 0.0,
                })

    df = pd.DataFrame(rows)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    # Remove duplicate timestamps
    df = df.drop_duplicates(subset='timestamp_utc', keep='first').reset_index(drop=True)
    return df


def compute_sog_cog(df):
    """Compute SOG (knots) and COG (degrees) from consecutive points."""
    sog = [0.0]
    cog = [0.0]

    for i in range(1, len(df)):
        dt = (df.loc[i, 'timestamp_utc'] - df.loc[i - 1, 'timestamp_utc']).total_seconds()
        if dt <= 0:
            sog.append(0.0)
            cog.append(cog[-1])
            continue

        dist = haversine_nm(
            df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
            df.loc[i, 'lat'], df.loc[i, 'lon']
        )
        sog.append(dist / dt * 3600)  # nm/s -> knots
        cog.append(bearing(
            df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
            df.loc[i, 'lat'], df.loc[i, 'lon']
        ))

    df['SOG'] = sog
    df['COG'] = cog
    return df


def filter_outliers(df, max_speed_knots):
    """Remove GPS jumps where instantaneous speed exceeds threshold."""
    mask = df['SOG'] <= max_speed_knots
    mask.iloc[0] = True  # keep first point
    removed = (~mask).sum()
    if removed > 0:
        print(f"  Filtered {removed} outlier points (SOG > {max_speed_knots} kn)")
    df = df[mask].reset_index(drop=True)
    # Recompute SOG/COG after filtering
    df = compute_sog_cog(df)
    return df


def resample_gaps(df, gap_threshold_sec, interval_sec):
    """Fill large gaps with linearly interpolated points."""
    rows = []
    for i in range(len(df)):
        rows.append(df.iloc[i].to_dict())
        if i < len(df) - 1:
            dt = (df.loc[i + 1, 'timestamp_utc'] - df.loc[i, 'timestamp_utc']).total_seconds()
            if dt > gap_threshold_sec:
                n_fill = int(dt / interval_sec) - 1
                for j in range(1, n_fill + 1):
                    frac = j / (n_fill + 1)
                    ts = df.loc[i, 'timestamp_utc'] + pd.Timedelta(seconds=interval_sec * j)
                    rows.append({
                        'timestamp_utc': ts,
                        'lat': df.loc[i, 'lat'] + frac * (df.loc[i + 1, 'lat'] - df.loc[i, 'lat']),
                        'lon': df.loc[i, 'lon'] + frac * (df.loc[i + 1, 'lon'] - df.loc[i, 'lon']),
                        'ele': df.loc[i, 'ele'] + frac * (df.loc[i + 1, 'ele'] - df.loc[i, 'ele']),
                        'SOG': np.nan,  # will be recomputed
                        'COG': np.nan,
                    })

    result = pd.DataFrame(rows).sort_values('timestamp_utc').reset_index(drop=True)
    # Recompute SOG/COG for the full resampled dataset
    result = compute_sog_cog(result)
    interpolated = len(result) - len(df)
    if interpolated > 0:
        print(f"  Interpolated {interpolated} points in gaps > {gap_threshold_sec}s")
    return result


def run(config):
    """Full parse pipeline: GPX -> SOG/COG -> filter -> resample."""
    gpx_cfg = config['gpx']
    print("[Step 1] Parsing GPX...")

    df = parse_gpx(gpx_cfg['file'])
    print(f"  Loaded {len(df)} trackpoints")
    print(f"  Time range: {df['timestamp_utc'].iloc[0]} ~ {df['timestamp_utc'].iloc[-1]}")

    df = compute_sog_cog(df)
    df = filter_outliers(df, gpx_cfg['max_speed_knots'])
    df = resample_gaps(df, gpx_cfg['resample_gap_sec'], gpx_cfg['resample_interval_sec'])

    print(f"  Final: {len(df)} points")
    print(f"  SOG range: {df['SOG'].min():.1f} ~ {df['SOG'].max():.1f} kn")
    return df
