import os
import warnings
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Configuration & Directories
# --------------------------------------------------
OS_DIRS = ["./image", "./image_rain"]
for d in OS_DIRS:
    os.makedirs(d, exist_ok=True)

# Yesterday 00 UTC initialization
init_date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
    hour=0, minute=0, second=0, microsecond=0, tzinfo=None
)

# 5-day forecast at 3-hour intervals (0 to 120h = 41 timesteps)
forecast_hours = list(range(0, 121, 3))

# Malaysia bounding box
LON_SLICE = slice(99.5, 119.5)
LAT_SLICE = slice(8.0, 0.5)  # GFS latitude runs North to South


# --------------------------------------------------
# 2. Download & Assemble GFS Dataset
# --------------------------------------------------
print(f"Downloading 5-day GFS forecast (3-hourly) initialized at {init_date} UTC...")
datasets = []

for fxx in forecast_hours:
    print(f"  Fetching F{fxx:03d}...")
    H = Herbie(
        date=init_date,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
        verbose=False,
    )

    # F000 does not contain accumulated precipitation (APCP)
    search_str = (
        ":(TMP:2 m above ground|DPT:2 m above ground):"
        if fxx == 0
        else ":(TMP:2 m above ground|DPT:2 m above ground|APCP:surface):"
    )

    ds_step = H.xarray(search_str)
    if isinstance(ds_step, list):
        ds_step = xr.merge(ds_step, compat="override")

    # Subset Malaysia immediately to save memory
    ds_step = ds_step.sel(longitude=LON_SLICE, latitude=LAT_SLICE)

    # Ensure 'tp' exists at F000 as zeros
    if "tp" not in ds_step:
        ds_step["tp"] = xr.zeros_like(ds_step["t2m"])

    valid_time = init_date + timedelta(hours=fxx)
    ds_step = ds_step.expand_dims(time=[valid_time])
    datasets.append(ds_step[["t2m", "d2m", "tp"]])

ds = xr.concat(datasets, dim="time").sortby("time").compute()


# --------------------------------------------------
# 3. Meteorology Calculations (NOAA Heat Index)
# --------------------------------------------------
def calculate_rh(t_c, td_c):
    """Magnus-Tetens approximation for Relative Humidity (%)."""
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    return np.clip(100.0 * (e / es), 0.0, 100.0)


def calculate_noaa_heat_index(t_f, rh):
    """Official NOAA / NWS Heat Index algorithm in Fahrenheit."""
    # Steadman simple formula
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    avg_hi = 0.5 * (hi_simple + t_f)

    # Full Rothfusz regression
    hi_full = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * (t_f**2)
        - 5.481717e-2 * (rh**2)
        + 1.22874e-3 * (t_f**2) * rh
        + 8.5282e-4 * t_f * (rh**2)
        - 1.99e-6 * (t_f**2) * (rh**2)
    )

    # Low-humidity adjustment (RH < 13% and 80 <= T <= 112)
    cond_low = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    adj_low = ((13.0 - rh) / 4.0) * np.sqrt((17.0 - np.abs(t_f - 95.0)) / 17.0)
    hi_full = xr.where(cond_low, hi_full - adj_low, hi_full)

    # High-humidity adjustment (RH > 85% and 80 <= T <= 87)
    cond_high = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    adj_high = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi_full = xr.where(cond_high, hi_full + adj_high, hi_full)

    return xr.where(avg_hi < 80.0, hi_simple, hi_full)


t_c = ds["t2m"] - 273.15
td_c = ds["d2m"] - 273.15
t_f = t_c * 9.0 / 5.0 + 32.0
rh = calculate_rh(t_c, td_c)

HI = calculate_noaa_heat_index(t_f, rh)


# --------------------------------------------------
# 4. De-accumulate 3-Hourly Rainfall (GFS 6h Resets)
# --------------------------------------------------
# In GFS 3-hourly files:
# - Odd steps (F003, F009, F015...) hold 3-hour totals since the last 6h reset
# - Even steps (F006, F012, F018...) hold 6-hour totals since the previous even step
rain_3hr_list = []
for i in range(1, len(forecast_hours)):
    fxx = forecast_hours[i]
    if fxx % 6 == 3:
        # Already a 3-hour accumulation
        step_rain = ds["tp"].isel(time=i)
    else:
        # Subtract prior 3-hour step from the 6-hour accumulation
        step_rain = ds["tp"].isel(time=i) - ds["tp"].isel(time=i - 1)
    rain_3hr_list.append(step_rain.clip(min=0.0))

rain_3hr = xr.concat(rain_3hr_list, dim="time")


# --------------------------------------------------
# 5. Plotting Helper & Map Generation
# --------------------------------------------------
def setup_map_ax(figsize=(8, 4.5)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([99.5, 119.5, 0.5, 8.0], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    return fig, ax


# --- Plot A: NOAA Heat Index ---
hi_bounds = [80, 90, 103, 125, 150]
hi_colors = ["#ffe082", "#ffb74d", "#e64a19", "#880e4f"]
hi_cmap = mcolors.ListedColormap(hi_colors)
hi_norm = mcolors.BoundaryNorm(hi_bounds, hi_cmap.N)

print("Generating Heat Index maps...")
for idx in range(len(HI.time)):
    field = np.where(HI.isel(time=idx).values >= 80, HI.isel(time=idx).values, np.nan)
    valid_utc = pd.to_datetime(HI.time.values[idx])
    valid_myt = valid_utc + timedelta(hours=8)

    fig, ax = setup_map_ax()
    im = ax.contourf(
        HI.longitude,
        HI.latitude,
        field,
        levels=hi_bounds,
        cmap=hi_cmap,
        norm=hi_norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(
        f"NOAA Heat Index Forecast | Init: {init_date:%Y-%m-%d %H:%M} UTC\n"
        f"Valid: {valid_myt:%Y-%m-%d %H:%M} MYT ({valid_utc:%H:%M} UTC)",
        fontsize=9.5,
    )
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.06, aspect=35, shrink=0.85, ax=ax)
    cbar.set_ticks([85, 96.5, 114, 137.5])
    cbar.set_ticklabels([
        "80–90°F: Caution",
        "90–103°F: Ext. Caution",
        "103–124°F: Danger",
        "≥125°F: Ext. Danger",
    ])
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(f"./image/Heat_index_F{forecast_hours[idx]:03d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Plot B: 3-Hourly Accumulated Rainfall ---
rain_levels = [0.1, 3, 6, 9, 12, 15, 20, 25, 30]
rain_cmap = plt.get_cmap("Blues", len(rain_levels) - 1)
rain_norm = mcolors.BoundaryNorm(rain_levels, rain_cmap.N)

print("Generating 3-Hourly Rainfall maps...")
for idx in range(len(rain_3hr.time)):
    raw_field = rain_3hr.isel(time=idx).values
    field = np.where(raw_field >= 0.1, raw_field, np.nan)
    valid_utc = pd.to_datetime(rain_3hr.time.values[idx])
    valid_myt = valid_utc + timedelta(hours=8)
    fxx = forecast_hours[idx + 1]

    fig, ax = setup_map_ax()
    im = ax.contourf(
        rain_3hr.longitude,
        rain_3hr.latitude,
        field,
        levels=rain_levels,
        cmap=rain_cmap,
        norm=rain_norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(
        f"3-Hourly Accumulated Rainfall (mm) | Init: {init_date:%Y-%m-%d %H:%M} UTC\n"
        f"Valid: {valid_myt:%Y-%m-%d %H:%M} MYT ({valid_utc:%H:%M} UTC)",
        fontsize=9.5,
    )
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.06, aspect=35, shrink=0.85, ax=ax)
    cbar.set_label("Rainfall (mm / 3 hr)", fontsize=8)
    cbar.set_ticks(rain_levels)
    cbar.set_ticklabels([str(x) for x in rain_levels])
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(f"./image_rain/Rainfall_F{fxx:03d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Done! All maps saved.")
