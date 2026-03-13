"""Noise injection, gust simulation, and gap/spike generation."""

import numpy as np
import pandas as pd


def add_gaussian_noise(df, aws_sigma, awa_sigma):
    """Add Gaussian noise to AWS and AWA."""
    n = len(df)
    df['AWS'] = df['AWS'] + np.random.normal(0, aws_sigma, n)
    df['AWA'] = df['AWA'] + np.random.normal(0, awa_sigma, n)
    # Clamp AWS to non-negative
    df['AWS'] = df['AWS'].clip(lower=0)
    # Normalize AWA to -180..+180
    df['AWA'] = (df['AWA'] + 180) % 360 - 180
    return df


def add_gusts(df, config):
    """Simulate wind gusts as temporary AWS boosts."""
    gust_cfg = config['noise']['gust']
    total_hours = (df['timestamp_utc'].iloc[-1] - df['timestamp_utc'].iloc[0]).total_seconds() / 3600
    n_gusts = np.random.poisson(gust_cfg['avg_per_hour'] * total_hours)

    events = []
    t_start = df['timestamp_utc'].iloc[0]
    t_end = df['timestamp_utc'].iloc[-1]
    total_sec = (t_end - t_start).total_seconds()

    for _ in range(n_gusts):
        onset = t_start + pd.Timedelta(seconds=np.random.uniform(0, total_sec))
        duration = np.random.uniform(gust_cfg['min_duration_sec'], gust_cfg['max_duration_sec'])
        boost = np.random.uniform(gust_cfg['min_boost_knots'], gust_cfg['max_boost_knots'])
        end_time = onset + pd.Timedelta(seconds=duration)

        mask = (df['timestamp_utc'] >= onset) & (df['timestamp_utc'] <= end_time)
        if mask.any():
            # Ramp-up / ramp-down envelope
            gust_times = df.loc[mask, 'timestamp_utc']
            center = onset + pd.Timedelta(seconds=duration / 2)
            half_dur = duration / 2
            envelope = 1 - np.abs((gust_times - center).dt.total_seconds().values / half_dur)
            envelope = np.clip(envelope, 0, 1)
            df.loc[mask, 'AWS'] = df.loc[mask, 'AWS'] + boost * envelope

            events.append({
                'timestamp_utc': onset,
                'event_type': 'GUST',
                'duration_sec': round(duration, 1),
                'description': f'AWS boost +{boost:.1f}kn',
            })

    print(f"  Added {n_gusts} gust events")
    return df, events


def inject_gaps_and_spikes(df, config):
    """Inject SHORT gaps, SENSOR OFFLINE periods, and SPIKE values."""
    gap_cfg = config['noise']['gaps']
    total_hours = (df['timestamp_utc'].iloc[-1] - df['timestamp_utc'].iloc[0]).total_seconds() / 3600

    # Initialize quality column
    df['quality'] = config['quality']['normal']

    events = []
    t_start = df['timestamp_utc'].iloc[0]
    t_end = df['timestamp_utc'].iloc[-1]
    total_sec = (t_end - t_start).total_seconds()

    # SHORT gaps
    n_short = np.random.poisson(gap_cfg['short']['per_hour'] * total_hours)
    for _ in range(n_short):
        onset = t_start + pd.Timedelta(seconds=np.random.uniform(0, total_sec))
        duration = np.random.uniform(gap_cfg['short']['min_sec'], gap_cfg['short']['max_sec'])
        end_time = onset + pd.Timedelta(seconds=duration)

        mask = (df['timestamp_utc'] >= onset) & (df['timestamp_utc'] <= end_time)
        df.loc[mask, 'AWS'] = np.nan
        df.loc[mask, 'AWA'] = np.nan
        df.loc[mask, 'quality'] = config['quality']['missing']

        events.append({
            'timestamp_utc': onset,
            'event_type': 'SHORT_GAP',
            'duration_sec': round(duration, 1),
            'description': f'{mask.sum()} points nulled',
        })

    # SENSOR OFFLINE
    n_offline = np.random.poisson(gap_cfg['offline']['per_hour'] * total_hours)
    for _ in range(n_offline):
        onset = t_start + pd.Timedelta(seconds=np.random.uniform(0, total_sec))
        duration = np.random.uniform(gap_cfg['offline']['min_sec'], gap_cfg['offline']['max_sec'])
        end_time = onset + pd.Timedelta(seconds=duration)

        mask = (df['timestamp_utc'] >= onset) & (df['timestamp_utc'] <= end_time)
        df.loc[mask, 'AWS'] = np.nan
        df.loc[mask, 'AWA'] = np.nan
        df.loc[mask, 'quality'] = config['quality']['missing']

        events.append({
            'timestamp_utc': onset,
            'event_type': 'SENSOR_OFFLINE',
            'duration_sec': round(duration, 1),
            'description': f'{mask.sum()} points nulled',
        })

    # SPIKES
    n_spike = np.random.poisson(gap_cfg['spike']['per_hour'] * total_hours)
    valid_idx = df.index[df['quality'] == config['quality']['normal']].tolist()
    if len(valid_idx) > 0:
        spike_indices = np.random.choice(valid_idx, size=min(n_spike, len(valid_idx)), replace=False)
        for idx in spike_indices:
            multiplier = np.random.uniform(gap_cfg['spike']['min_multiplier'],
                                           gap_cfg['spike']['max_multiplier'])
            original_aws = df.loc[idx, 'AWS']
            df.loc[idx, 'AWS'] = original_aws * multiplier
            df.loc[idx, 'AWA'] = np.random.uniform(-180, 180)
            df.loc[idx, 'quality'] = config['quality']['suspect']

            events.append({
                'timestamp_utc': df.loc[idx, 'timestamp_utc'],
                'event_type': 'SPIKE',
                'duration_sec': 0,
                'description': f'AWS {original_aws:.1f} -> {df.loc[idx, "AWS"]:.1f}kn',
            })

    print(f"  Injected: {n_short} short gaps, {n_offline} offline periods, {len(spike_indices) if len(valid_idx) > 0 else 0} spikes")
    return df, events


def run(config, df):
    """Full noise pipeline."""
    print("[Step 4] Noise and gap injection...")
    np.random.seed(42)  # reproducible

    df = add_gaussian_noise(df, config['noise']['aws_sigma'], config['noise']['awa_sigma'])
    df, gust_events = add_gusts(df, config)
    df, gap_events = inject_gaps_and_spikes(df, config)

    all_events = gust_events + gap_events
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = events_df.sort_values('timestamp_utc').reset_index(drop=True)

    total_missing = (df['quality'] == config['quality']['missing']).sum()
    total_suspect = (df['quality'] == config['quality']['suspect']).sum()
    print(f"  Quality summary: {len(df) - total_missing - total_suspect} normal, "
          f"{total_missing} missing, {total_suspect} suspect")
    return df, events_df
