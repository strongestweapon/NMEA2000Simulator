#!/usr/bin/env python3
"""NMEA 2000 Wind Data Simulator - Pipeline Runner.

Usage:
    python main.py --step parse        # GPX -> SOG/COG
    python main.py --step era5         # ERA5 download + interpolation
    python main.py --step correct      # Polar inversion + Kalman wind correction
    python main.py --step calc         # AWS/AWA calculation
    python main.py --step noise        # Noise + gap injection
    python main.py --step validate     # Validation plots
    python main.py --step all          # Run full pipeline
    python main.py --stream --speed 10 # Stream to ESP32
"""

import argparse
import pickle
from pathlib import Path

import yaml


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


CACHE_DIR = Path('data/output/.cache')


def save_state(name, obj):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_state(name):
    path = CACHE_DIR / f'{name}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def step_parse(config):
    from src.parser import run
    df = run(config)
    save_state('parsed', df)
    return df


def step_era5(config, df=None):
    if df is None:
        df = load_state('parsed')
        if df is None:
            raise RuntimeError("Run 'parse' step first")
    from src.era5 import run
    df = run(config, df)
    save_state('era5', df)
    return df


def step_correct(config, df=None):
    if df is None:
        df = load_state('era5')
        if df is None:
            raise RuntimeError("Run 'era5' step first")
    from src.polar import run
    df = run(config, df)
    save_state('corrected', df)
    return df


def step_calc(config, df=None):
    if df is None:
        df = load_state('corrected')
        if df is None:
            df = load_state('era5')
        if df is None:
            raise RuntimeError("Run 'era5' step first")
    from src.wind_calc import run
    df = run(config, df)
    save_state('calc', df)
    return df


def step_noise(config, df=None):
    if df is None:
        df = load_state('calc')
        if df is None:
            raise RuntimeError("Run 'calc' step first")
    from src.noise import run
    df, events_df = run(config, df)
    save_state('noise', (df, events_df))
    return df, events_df


def step_output(config, df=None, events_df=None):
    if df is None:
        state = load_state('noise')
        if state is None:
            raise RuntimeError("Run 'noise' step first")
        df, events_df = state
    from src.output import run
    run(config, df, events_df)
    return df, events_df


def step_validate(config, df=None, events_df=None):
    if df is None:
        state = load_state('noise')
        if state is None:
            raise RuntimeError("Run 'noise' step first")
        df, events_df = state
    from src.visualize import run
    run(config, df, events_df)


def run_all(config):
    df = step_parse(config)
    df = step_era5(config, df)
    df = step_correct(config, df)
    df = step_calc(config, df)
    df, events_df = step_noise(config, df)
    step_output(config, df, events_df)
    step_validate(config, df, events_df)
    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='NMEA 2000 Wind Data Simulator')
    parser.add_argument('--step', choices=['parse', 'era5', 'correct', 'calc', 'noise', 'output', 'validate', 'all'],
                        default='all', help='Pipeline step to run')
    parser.add_argument('--stream', action='store_true', help='Stream data to serial/stdout')
    parser.add_argument('--speed', type=float, default=1, help='Playback speed multiplier')
    parser.add_argument('--port', type=str, default=None, help='Serial port (e.g. /dev/tty.usbserial)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stream:
        state = load_state('noise')
        if state is None:
            raise RuntimeError("Run pipeline first before streaming")
        df, _ = state
        from src.output import stream_serial
        stream_serial(df, config, speed=args.speed, port=args.port)
        return

    steps = {
        'parse': lambda: step_parse(config),
        'era5': lambda: step_era5(config),
        'correct': lambda: step_correct(config),
        'calc': lambda: step_calc(config),
        'noise': lambda: step_noise(config),
        'output': lambda: step_output(config),
        'validate': lambda: step_validate(config),
        'all': lambda: run_all(config),
    }

    steps[args.step]()


if __name__ == '__main__':
    main()
