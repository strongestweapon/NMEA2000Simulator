"""ERA5 wind data download and spatiotemporal interpolation."""

import os
from datetime import timedelta
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc


def _build_request(config, df):
    """Build CDS API request parameters from track data."""
    era5_cfg = config['era5']
    buf = era5_cfg['time_buffer_hours']

    t_min = df['timestamp_utc'].min() - timedelta(hours=buf)
    t_max = df['timestamp_utc'].max() + timedelta(hours=buf)

    # Collect unique dates and hours
    dates = pd.date_range(t_min.floor('D'), t_max.ceil('D'), freq='D')
    days = sorted(set(d.strftime('%d') for d in dates))
    months = sorted(set(d.strftime('%m') for d in dates))
    years = sorted(set(d.strftime('%Y') for d in dates))
    hours = [f"{h:02d}:00" for h in range(24)]

    request = {
        'product_type': ['reanalysis'],
        'variable': era5_cfg['variables'],
        'year': years,
        'month': months,
        'day': days,
        'time': hours,
        'data_format': 'netcdf',
        'area': [
            era5_cfg['area']['north'],
            era5_cfg['area']['west'],
            era5_cfg['area']['south'],
            era5_cfg['area']['east'],
        ],
    }
    return request


def download_era5(config, df):
    """Download ERA5 wind data, using cache if available."""
    era5_cfg = config['era5']
    cache_dir = Path(era5_cfg['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'era5_wind.nc'

    if cache_file.exists():
        print(f"  Using cached ERA5 data: {cache_file}")
        return str(cache_file)

    print("  Downloading ERA5 data (this may take a few minutes)...")
    request = _build_request(config, df)

    client = cdsapi.Client()
    client.retrieve(era5_cfg['dataset'], request, str(cache_file))
    print(f"  Saved to {cache_file}")
    return str(cache_file)


def load_era5(nc_path):
    """Load ERA5 netCDF and return time/lat/lon grids + U/V arrays."""
    ds = nc.Dataset(nc_path, 'r')

    # Time: hours since epoch or similar
    time_var = ds.variables['valid_time'] if 'valid_time' in ds.variables else ds.variables['time']
    time_units = time_var.units
    time_calendar = getattr(time_var, 'calendar', 'standard')
    times = nc.num2date(time_var[:], units=time_units, calendar=time_calendar)
    # cftime objects -> epoch seconds (handle cftime.DatetimeProlepticGregorian)
    times_epoch = np.array([
        pd.Timestamp(str(t)).timestamp() for t in times
    ])

    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]

    # Variable names vary between CDS versions
    u_name = 'u10' if 'u10' in ds.variables else 'u_component_of_wind_10m'
    v_name = 'v10' if 'v10' in ds.variables else 'v_component_of_wind_10m'
    u10 = ds.variables[u_name][:]  # (time, lat, lon)
    v10 = ds.variables[v_name][:]

    ds.close()

    # Ensure lat is ascending for interpolation
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u10 = u10[:, ::-1, :]
        v10 = v10[:, ::-1, :]

    return times_epoch, np.array(lat), np.array(lon), np.array(u10), np.array(v10)


def interpolate_wind(df, nc_path):
    """Interpolate ERA5 U/V to each GPS point (time + lat/lon)."""
    times_epoch, lat, lon, u10, v10 = load_era5(nc_path)

    # Build 3D interpolators (time, lat, lon)
    interp_u = RegularGridInterpolator(
        (times_epoch, lat, lon), u10,
        method='linear', bounds_error=False, fill_value=None
    )
    interp_v = RegularGridInterpolator(
        (times_epoch, lat, lon), v10,
        method='linear', bounds_error=False, fill_value=None
    )

    # Query points
    pts_epoch = df['timestamp_utc'].apply(lambda t: t.timestamp()).values
    pts = np.column_stack([pts_epoch, df['lat'].values, df['lon'].values])

    u_interp = interp_u(pts)
    v_interp = interp_v(pts)

    # Convert U/V (m/s) to TWS (knots) and TWD (degrees, meteorological FROM)
    ms_to_knots = 1.94384
    tws = np.sqrt(u_interp**2 + v_interp**2) * ms_to_knots
    twd = (270 - np.degrees(np.arctan2(v_interp, u_interp))) % 360

    df['TWS'] = tws
    df['TWD'] = twd
    return df


def run(config, df):
    """Full ERA5 pipeline: download + interpolate."""
    print("[Step 2] ERA5 wind data...")
    nc_path = download_era5(config, df)
    df = interpolate_wind(df, nc_path)
    print(f"  TWS range: {df['TWS'].min():.1f} ~ {df['TWS'].max():.1f} kn")
    print(f"  TWD range: {df['TWD'].min():.1f} ~ {df['TWD'].max():.1f}°")
    return df
