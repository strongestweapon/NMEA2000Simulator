"""CSV output and serial streaming for ESP32."""

import json
import time
from pathlib import Path

import pandas as pd


def write_csv(df, config):
    """Write simulated wind data CSV."""
    out_path = Path(config['output']['csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ['timestamp_utc', 'lat', 'lon', 'SOG', 'COG', 'TWS', 'TWD', 'AWS', 'AWA', 'quality']
    out_df = df[cols].copy()

    # Round numeric columns
    for col in ['lat', 'lon']:
        out_df[col] = out_df[col].round(6)
    for col in ['SOG', 'COG', 'TWS', 'TWD', 'AWS', 'AWA']:
        out_df[col] = out_df[col].round(2)
    out_df['quality'] = out_df['quality'].astype(int)

    out_df.to_csv(out_path, index=False)
    print(f"  Written: {out_path} ({len(out_df)} rows)")
    return out_path


def write_events(events_df, config):
    """Write events log CSV."""
    out_path = Path(config['output']['events'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if events_df.empty:
        events_df = pd.DataFrame(columns=['timestamp_utc', 'event_type', 'duration_sec', 'description'])

    events_df.to_csv(out_path, index=False)
    print(f"  Written: {out_path} ({len(events_df)} events)")
    return out_path


def stream_serial(df, config, speed=1, port=None):
    """Stream wind data as JSON over serial or stdout.

    Format: {"ts":1741234567,"aws":12.3,"awa":-35.2,"q":1}
    """
    if port:
        import serial
        ser = serial.Serial(port, config['stream']['baudrate'])
        send = lambda msg: ser.write((msg + '\n').encode())
    else:
        send = lambda msg: print(msg)

    valid = df.dropna(subset=['AWS', 'AWA']).copy()
    valid['epoch'] = valid['timestamp_utc'].apply(lambda t: int(t.timestamp()))

    print(f"\n  Streaming {len(valid)} data points at {speed}x speed...")
    print("  Press Ctrl+C to stop\n")

    try:
        for i in range(len(valid)):
            row = valid.iloc[i]
            msg = json.dumps({
                'ts': int(row['epoch']),
                'aws': round(row['AWS'], 1),
                'awa': round(row['AWA'], 1),
                'q': int(row['quality']),
            })
            send(msg)

            if i < len(valid) - 1:
                dt = (valid.iloc[i + 1]['timestamp_utc'] - row['timestamp_utc']).total_seconds()
                wait = max(0, dt / speed)
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("\n  Streaming stopped.")
    finally:
        if port:
            ser.close()


def run(config, df, events_df):
    """Write all output files."""
    print("[Step 5] Writing output...")
    write_csv(df, config)
    write_events(events_df, config)
