import pandas as pd
import xarray as xr
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
os.makedirs('./image_rain', exist_ok=True)
os.makedirs('./image', exist_ok=True)

ct=datetime.datetime.now().date()

def subtract_days_from_date(date,days):
    subtracted_date=pd.to_datetime(date)-timedelta(days=days)
    subtracted_date=subtracted_date.strftime("%d-%m-%Y")

    return subtracted_date

adate=subtract_days_from_date(ct,1)

from datetime import datetime
adate1=datetime.strptime(adate, "%d-%m-%Y")
yyyy = adate1.strftime("%Y")
mm = adate1.strftime("%m")
dd = adate1.strftime("%d")

url=f'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{yyyy}{mm}{dd}/gfs_0p25_00z'

ds = xr.open_dataset(url, engine='netcdf4')

ds_Malaysian=ds.sel(lon=slice(100,120),lat=slice(0,12))

def temp2F(ds):
    celcius = ds - 273.15
    fahrenheit = (celcius*(9/5))+32
    return fahrenheit

def calculate_heat_index(temperature_f, relative_humidity_percent):
    """
    Calculates the Heat Index using the NWS simplified regression equation.
    """
    rh = relative_humidity_percent
    hi = (-42.379 + 2.04901523 * temperature_f + 10.14333127 * rh -
          0.22475541 * temperature_f * rh - 6.83783e-3 * temperature_f**2 -
          5.481717e-2 * rh**2 + 1.22874e-3 * temperature_f**2 * rh +
          8.5282e-4 * temperature_f * rh**2 - 1.99e-6 * temperature_f**2 * rh**2)
    return hi

tmp_2m_f=temp2F(ds_Malaysian['tmp2m'])

rh_2m=ds_Malaysian['rh2m']

HI=calculate_heat_index(tmp_2m_f,rh_2m)

HI_computed=HI.compute()

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

longitude=HI_computed.lon
latitude=HI_computed.lat
data=HI_computed.values
timestep=HI_computed.time

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
    ax.set_title(f"Forecast heat index, init: {str(adate)}, \n valid: {str(formatted_time)}", fontsize=10)
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

rain = ds_Malaysian['apcpsfc']  # kg m-2 == mm

# Compute 3-hourly rainfall (difference between time steps)
rain_3hr = rain.diff(dim='time')  # Now has one less time step

# Adjust time (optional, for plotting)
rain_time = ds_Malaysian['time'][1:]  # Matches dimensions after diff

# Compute to load into memory
rain_3hr_computed = rain_3hr.compute()

# Mask values <0.1 mm to avoid log(0)
field = np.where(field < 0.1, np.nan, field)

# Define a custom colormap that transitions from white to rainbow
blue = cm.get_cmap('Blues')  # Accessing rainbow colormap from cm
# Create the colormap from white to blue
white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"], N=256)

# create a custom boundary for levels
level = np.logspace(np.log10(0.1), np.log10(50), num=10)

# Creat boundaries such as anything below 1 will be mapped to white
boundaries = np.concatenate([[-np.inf], level[1:], [np.inf]])

norm=BoundaryNorm(boundaries, ncolors=256)

for idx in range(41):
    field = rain_3hr_computed[idx]
    forecast_time = timestep[idx]
    formatted_time = pd.to_datetime(forecast_time.values).strftime('%Y-%m-%d %H:%M')

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(longitude, latitude, field, transform=ccrs.PlateCarree(), cmap='Blues', levels=level, norm=norm,extend='max')
    ax.coastlines()
    ax.set_title(f"3-hourly Accumulated Rainfall (mm), init: {adate}, \n valid: {formatted_time}", fontsize=10)
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.02, aspect=30, shrink=0.8, ax=ax, location='bottom')
    cbar.set_label("3-hourly Accumulated Rainfall (mm)", fontsize='8')
    cbar.ax.tick_params(labelsize=6)
    plt.savefig(f'./image_rain/Rainfall_map_{idx}.png', dpi=300)
    plt.close()
