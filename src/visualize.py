"""Validation visualizations: route map, wind timeseries, wind rose, gaps timeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from windrose import WindroseAxes


def _ensure_dir(config):
    d = Path(config['output']['validation_dir'])
    d.mkdir(parents=True, exist_ok=True)
    return d


def route_map(df, config):
    """Interactive folium map with GPS track colored by SOG."""
    out_dir = _ensure_dir(config)
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles='cartodbpositron')

    # Color track by SOG
    coords = list(zip(df['lat'], df['lon']))
    sog = df['SOG'].values
    sog_norm = np.clip(sog / 15, 0, 1)  # normalize to 0-15 kn

    # Draw segments
    for i in range(1, len(coords)):
        intensity = sog_norm[i]
        r = int(255 * intensity)
        b = int(255 * (1 - intensity))
        color = f'#{r:02x}40{b:02x}'
        folium.PolyLine(
            [coords[i-1], coords[i]],
            color=color, weight=2, opacity=0.8
        ).add_to(m)

    # Start/end markers
    folium.Marker(coords[0], popup='START: Hong Kong',
                  icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(coords[-1], popup='FINISH: Subic Bay',
                  icon=folium.Icon(color='red')).add_to(m)

    out_path = out_dir / 'route_map.html'
    m.save(str(out_path))
    print(f"  Saved: {out_path}")


def wind_timeseries(df, config):
    """TWS/TWD/AWS/AWA time series plot."""
    out_dir = _ensure_dir(config)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Wind Data Time Series - Lisa Elaine RCSR 2026', fontsize=14, y=0.98)

    times = df['timestamp_utc']

    # Define leg boundaries (approximate)
    hk_end = pd.Timestamp('2026-03-04T08:00:00', tz='UTC')    # ~16:00 HKT
    subic_start = pd.Timestamp('2026-03-07T02:00:00', tz='UTC')  # ~10:00 HKT

    def color_by_leg(t):
        colors = []
        for ts in t:
            if ts <= hk_end:
                colors.append('#e74c3c')    # HK nearshore - red
            elif ts >= subic_start:
                colors.append('#2ecc71')    # Subic nearshore - green
            else:
                colors.append('#3498db')    # Open ocean - blue
        return colors

    colors = color_by_leg(times)

    # TWS
    ax = axes[0]
    ax.scatter(times, df['TWS'], c=colors, s=1, alpha=0.6)
    ax.set_ylabel('TWS (kn)')
    ax.set_ylim(0, df['TWS'].quantile(0.99) * 1.3)
    ax.grid(True, alpha=0.3)

    # TWD
    ax = axes[1]
    ax.scatter(times, df['TWD'], c=colors, s=1, alpha=0.6)
    ax.set_ylabel('TWD (°)')
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.3)

    # AWS
    ax = axes[2]
    valid = df['quality'] != config['quality']['missing']
    suspect = df['quality'] == config['quality']['suspect']
    ax.scatter(times[valid & ~suspect], df.loc[valid & ~suspect, 'AWS'],
               c=[c for c, v in zip(colors, valid & ~suspect) if v], s=1, alpha=0.6)
    ax.scatter(times[suspect], df.loc[suspect, 'AWS'], c='red', s=10, marker='x', label='Suspect')
    ax.set_ylabel('AWS (kn)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # AWA
    ax = axes[3]
    ax.scatter(times[valid & ~suspect], df.loc[valid & ~suspect, 'AWA'],
               c=[c for c, v in zip(colors, valid & ~suspect) if v], s=1, alpha=0.6)
    ax.scatter(times[suspect], df.loc[suspect, 'AWA'], c='red', s=10, marker='x')
    ax.set_ylabel('AWA (°)')
    ax.set_ylim(-180, 180)
    ax.set_xlabel('Time (UTC)')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate()

    # Legend for leg colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='HK nearshore'),
        Patch(facecolor='#3498db', label='Open ocean'),
        Patch(facecolor='#2ecc71', label='Subic nearshore'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = out_dir / 'wind_timeseries.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def wind_rose(df, config):
    """Wind rose plots split by leg."""
    out_dir = _ensure_dir(config)

    hk_end = pd.Timestamp('2026-03-04T08:00:00', tz='UTC')
    subic_start = pd.Timestamp('2026-03-07T02:00:00', tz='UTC')

    legs = [
        ('HK Nearshore', df[df['timestamp_utc'] <= hk_end]),
        ('Open Ocean', df[(df['timestamp_utc'] > hk_end) & (df['timestamp_utc'] < subic_start)]),
        ('Subic Nearshore', df[df['timestamp_utc'] >= subic_start]),
    ]

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('True Wind Rose by Leg - Lisa Elaine RCSR 2026', fontsize=14)

    for i, (title, leg_df) in enumerate(legs):
        valid = leg_df.dropna(subset=['TWS', 'TWD'])
        if len(valid) == 0:
            continue

        ax = WindroseAxes.from_ax(fig=fig, rect=[i * 0.33 + 0.02, 0.05, 0.28, 0.85])
        ax.bar(valid['TWD'].values, valid['TWS'].values,
               normed=True, opening=0.8, edgecolor='white', linewidth=0.3,
               bins=np.arange(0, valid['TWS'].max() + 5, 5))
        ax.set_title(title, fontsize=11, pad=15)
        ax.set_legend(loc='lower left', fontsize=7)

    out_path = out_dir / 'wind_rose.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def gaps_timeline(df, events_df, config):
    """Timeline showing injected gaps, spikes, and gusts."""
    out_dir = _ensure_dir(config)

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.suptitle('Injected Events Timeline', fontsize=14)

    if events_df.empty:
        ax.text(0.5, 0.5, 'No events', transform=ax.transAxes, ha='center')
        out_path = out_dir / 'gaps_timeline.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    event_colors = {
        'SHORT_GAP': '#e74c3c',
        'SENSOR_OFFLINE': '#c0392b',
        'SPIKE': '#f39c12',
        'GUST': '#3498db',
    }

    y_levels = {
        'SHORT_GAP': 1,
        'SENSOR_OFFLINE': 2,
        'SPIKE': 3,
        'GUST': 4,
    }

    for _, event in events_df.iterrows():
        etype = event['event_type']
        t = pd.Timestamp(event['timestamp_utc'])
        dur = event['duration_sec']
        color = event_colors.get(etype, 'gray')
        y = y_levels.get(etype, 0)

        if dur > 0:
            ax.barh(y, dur / 3600, left=mdates.date2num(t),
                    height=0.6, color=color, alpha=0.7)
        else:
            ax.scatter(mdates.date2num(t), y, c=color, s=20, zorder=5)

    ax.set_yticks(list(y_levels.values()))
    ax.set_yticklabels(list(y_levels.keys()))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate()
    ax.set_xlabel('Time (UTC)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(0.3, 4.7)

    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=k) for k, c in event_colors.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = out_dir / 'gaps_timeline.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run(config, df, events_df):
    """Generate all validation visualizations."""
    print("[Step 6] Generating validation plots...")
    route_map(df, config)
    wind_timeseries(df, config)
    wind_rose(df, config)
    gaps_timeline(df, events_df, config)
