import pandas as pd
import xarray as xr
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
os.makedirs('./image_rain', exist_ok=True)
os.makedirs('./image', exist_ok=True)

# --------------------------------------------------
# Initialization Time (Yesterday 00 UTC)
# --------------------------------------------------
init_date = datetime.datetime.utcnow().replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=1)

# --------------------------------------------------
# FULL GFS Forecast Hours (0–384)
# 0–120 hourly
# 126–384 every 3 hours
# --------------------------------------------------
forecast_hours = list(range(0, 121)) + list(range(126, 385, 3))

datasets = []

print("Downloading GFS full forecast...")

for fxx in forecast_hours:
    print(f"Downloading F{fxx:03d}")

    H = Herbie(
        date=init_date,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
    )

    ds = H.xarray(
        ":(TMP:2 m above ground|DPT:2 m above ground|APCP:surface):"
    )

    if isinstance(ds, list):
        ds = xr.merge(ds)

    valid_time = init_date + timedelta(hours=fxx)
    ds = ds.expand_dims(time=[valid_time])  # ensures each forecast hour has its own time

    datasets.append(ds)

# --------------------------------------------------
# Combine all timesteps
# --------------------------------------------------
ds = xr.concat(datasets, dim="time")
ds = ds.sortby("time")  # ensures time is ascending

print("Total timesteps:", len(ds.time))

# --------------------------------------------------
# Subset Malaysia
# --------------------------------------------------
ds = ds.sel(
    longitude=slice(100, 120),
    latitude=slice(12, 0)
)

# --------------------------------------------------
# Temperature Conversion
# --------------------------------------------------
def kelvin_to_fahrenheit(da):
    return (da - 273.15) * 9/5 + 32

tmp_f = kelvin_to_fahrenheit(ds["t2m"])

# --------------------------------------------------
# Relative Humidity Calculation
# --------------------------------------------------
T = ds["t2m"] - 273.15
Td = ds["d2m"] - 273.15

es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
e  = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))

rh = 100 * (e / es)

# --------------------------------------------------
# Heat Index Calculation
# --------------------------------------------------
def calculate_heat_index(T_f, RH):

    HI_simple = 0.5 * (
        T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094)
    )

    HI_full = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 6.83783e-3 * T_f**2
        - 5.481717e-2 * RH**2
        + 1.22874e-3 * T_f**2 * RH
        + 8.5282e-4 * T_f * RH**2
        - 1.99e-6 * T_f**2 * RH**2
    )

    return xr.where(T_f <= 80, HI_simple, HI_full)

HI = calculate_heat_index(tmp_f, rh).compute()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.colors import BoundaryNorm
import numpy as np

import matplotlib.colors as mcolors

# Define boundaries and labels
bounds = [80, 90,103,124]
labels = ["Caution", "Ext. Caution","Danger"]
colors = ['#ffe082', '#ffb74d', '#e64a19']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

### limit up to 5 days forecast, 40 timesteps

longitude = HI.longitude
latitude = HI.latitude
data = HI.values
timestep = HI.time

for idx in range(41):
    field = data[idx]
    forecast_time=timestep[idx]
    formatted_time = pd.to_datetime(forecast_time.values).strftime('%Y-%m-%d %H:%M')
    #masked out anything below 80F
    masked_field = np.where(field >= 80, field, np.nan)

    # Plot this timestep (simplified example)
    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    im = ax.contourf(longitude, latitude, masked_field, levels=bounds,transform=ccrs.PlateCarree(),cmap=cmap, norm=norm, extend='max')
    ax.coastlines()
    ax.set_title(f"Forecast heat index, init: {str(init_date)}, \n valid: {str(formatted_time)}", fontsize=10)
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.02, aspect=30, shrink=0.8, ax=ax, location='bottom')
    cbar.set_label("HI Risk Level",fontsize='8')
    #cbar.set_ticks([85, 96.5,113.5])  # approximate midpoints of the ranges
    #cbar.set_ticklabels(labels)
    cbar.set_ticks([85, 96.5, 113.5])
    cbar.set_ticklabels(["80–90: Caution", "90–103: Ext. Caution", "103–124: Danger"])
    cbar.ax.tick_params(labelsize=6) 
    plt.savefig('./image/Heat_index_init_'+str(idx)+'.png',dpi=300)
    plt.close()

### accumulated rainfall

rain = ds['apcp']  # kg m-2 == mm

# Compute 3-hourly rainfall (difference between time steps)
rain_3hr = rain.diff(dim='time')  # Now has one less time step

# Adjust time (optional, for plotting)
rain_time = ds['time'][1:]  # Matches dimensions after diff

# Compute to load into memory
rain_3hr_computed = rain_3hr.compute()

# Mask values <0.1 mm to avoid log(0)
#field = np.where(field < 0.1, np.nan, field)

# Define a custom colormap that transitions from white to rainbow
blue = cm.get_cmap('Blues')  # Accessing rainbow colormap from cm
# Create the colormap from white to blue
white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"], N=256)

# create a custom boundary for levels
level = [0,3,6,9,12,15,20,25,30]

# Creat boundaries such as anything below 1 will be mapped to white
boundaries = np.concatenate([[-np.inf], level[1:], [np.inf]])

# Create custom ticks
custom_ticks=[0, 3, 6, 9, 12, 15, 20, 25, 30]

norm=BoundaryNorm(boundaries, ncolors=256)

for idx in range(41):
    raw_field = rain_3hr_computed[idx]
    # Mask values <0.1 mm to avoid log(0)
    field = np.where(raw_field < 0.1, np.nan, raw_field)
    forecast_time = rain_time[idx]
    formatted_time = pd.to_datetime(forecast_time.values).strftime('%Y-%m-%d %H:%M')

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(longitude, latitude, field, transform=ccrs.PlateCarree(), cmap='Blues', levels=level, norm=norm,extend='max')
    ax.coastlines()
    ax.set_title(f"3-hourly Accumulated Rainfall (mm), init: {init_date}, \n valid: {formatted_time}", fontsize=10)
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.02, aspect=30, shrink=0.8, ax=ax, location='bottom')
    cbar.set_label("3-hourly Accumulated Rainfall (mm)", fontsize='8')
    cbar.set_ticks(custom_ticks)
    cbar.ax.tick_params(labelsize=6)
    plt.savefig(f'./image_rain/Rainfall_map_{idx}.png', dpi=300)
    plt.close()



