import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from scipy.stats import t
import glob
import os
import matplotlib as mpl


def SettingUpXarray(ModelNameDirection, VariableName):
    netCDF4File = xr.open_dataset(ModelNameDirection)
    LongitudeName = 'lon'
    LatitudeName = 'lat'
    
    #Correct Latitude-Longitude names
    while True:
        try:
            netCDF4File[LatitudeName]
            break
        except:
            print('Correcting wrong dimession names')
            print(list(netCDF4File.coords))
            # str(input('Correct longitude name: '))
            LongitudeName = 'longitude'
            LatitudeName = 'latitude'  # str(input('Correct latitude name: '))
            netCDF4File = netCDF4File.rename(
                {LatitudeName: 'lat', LongitudeName: 'lon'})
            LatitudeName = 'lat'
            LongitudeName = 'lon'

    #Correct variable names
    while True:
        try:
            netCDF4File[VariableName]
            break
        except:
            print('Wrong dimension names')
            print(netCDF4File.data_vars)
            VariableName = str(input('Correct variable name: '))
            netCDF4File = netCDF4File.rename({VariableName: 'pr'})
            VariableName = 'pr'
    #Verificar la existencia de unidades en el NC
    while True:
        try:
            netCDF4File[VariableName].units
            break
        except:
            print(netCDF4File.data_vars)
            print('Units not found')
            unit = str(input('Enter the units: '))
            netCDF4File[VariableName].attrs['units'] = unit

    #Unit conversion to mm/day
    if netCDF4File[VariableName].units == 'mm/hr':
        netCDF4File[VariableName] = 24*netCDF4File[VariableName]
    elif netCDF4File[VariableName].units == 'kg m-2 s-1':
        netCDF4File[VariableName] = 86400*netCDF4File[VariableName]
    elif netCDF4File[VariableName].units == 'mm':
        netCDF4File[VariableName] = (1/30)*netCDF4File[VariableName]

    return netCDF4File[VariableName]


def SlicenetCDFXarray(Precipitation, RegionToSlice):
    #if Precipitation.lat[0] < 0:
    SlicedRegion = Precipitation.where(
        (RegionToSlice[1, 0] <= Precipitation.lon) &
        (RegionToSlice[1, 1] >= Precipitation.lon) &
        (RegionToSlice[0, 0] < Precipitation.lat) &
        (RegionToSlice[0, 1] > Precipitation.lat), drop=True)
    return SlicedRegion


def CompositeCMIP(Precipitation, ONI, ENSO, *argv):
    ONI.index = pd.to_datetime(ONI.index)
    mask = (ONI.index > np.array(Precipitation['time'])[0]) & (
        ONI.index <= np.array(Precipitation['time'][-1] + pd.Timedelta(weeks=5)))
    ONI = ONI.loc[mask]
    #Compute composites for ENSO
    ENSOArray = np.where(np.array(ONI['Event'], dtype='U10') == ENSO)
    #Compute Long-Term composites
    ENSOComposite_Mean = Precipitation[list(
        ENSOArray)[0]].groupby('time.season').mean('time')
    ENSOComposite_STD = Precipitation[list(
        ENSOArray)[0]].groupby('time.season').std('time')
    # ENSOComposite_STD = None
    return ENSOComposite_Mean, ENSOComposite_STD, ENSOArray[0].shape[0]


def SeasonalPlot(Precipitation,axes,*args):
    #Loop into each season
    for k, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        cs = Precipitation.sel(season=season).plot.contourf(
            ax=axes[args[0],k], vmin=0, vmax=12,
            cmap='Spectral', add_colorbar=False,
            extend='max', transform=ccrs.PlateCarree())
        plt.colorbar(cs, ax=axes[args[0], k])
        #Set the aspect of the color bar to coincide with map
        axes[args[0], k].set_aspect('auto')
        #Add coastal lines
        axes[args[0], k].coastlines()
        #Delete title
        axes[args[0], k].set_title('')
        #Draw grid lines and set up the labels
        gl = axes[args[0], k].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                   linewidth=1, color='black', alpha=0.5, linestyle=':')
        gl.top_labels = False
        gl.bottom_labels = False
        gl.right_labels = False
        gl.left_labels = False
        if season == 'DJF':
            gl.left_labels = True
            axes[args[0], k].text(-0.35, 0.55, args[1], va='bottom', ha='center',
                                  rotation='vertical', rotation_mode='anchor',
                                  transform=axes[args[0], k].transAxes)
        #Place the titles
        if args[0] == 0:
            axes[args[0], k].set_title(season)
    pass

def main():
    InDirectoryCMIP5 = '/path/directory/historical/precipitation/cmip/nc_files'
    InDirectoryONIsCMIP5 = '/path/directory/oni/files/historical/for/each/CMIP_model'
    OutDirectoryFigures = '/path/dictory/where/results/will/be/saved/'
    #List files
    FileDirectionsCMIP = np.sort(np.array(glob.glob(InDirectoryCMIP5 + "*.nc")))
    FileDirectionsONIs = np.sort(np.array(glob.glob(InDirectoryONIsCMIP5 + "*.csv")))

    #Setting up figure canvas
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 5), gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                             sharex=True, sharey=True, subplot_kw=dict(projection=proj))
    ModelVarName = 'pr'
    RegionToPlot = np.array([[-10, 17],
                             [-100, -65]])
    #Model list
    ModelNames = ['GFDL-ESM2M','GISS-E2-H','MIROC5'] #Change name if necessary
    for i, ModelNameDirection in enumerate(FileDirectionsCMIP):
        print('Processing %s' % ModelNameDirection)
        print('Setting up the Xarray file')
        PrecipitationModel = SettingUpXarray(ModelNameDirection, 
                                             ModelVarName)
        #Fix the time in files
        SplittedName = ModelNameDirection.split('/')
        CMIPName = SplittedName[-1]
        #Begin, end dates for each file
        StartDate_MOD = CMIPName.split('_')[-1][0:4]
        EndDate_MOD = CMIPName.split('_')[-1][7:11]
        #FixTime
        print('Slicing time from %s to %s' % (StartDate_MOD, EndDate_MOD))
        Time = pd.date_range(start=StartDate_MOD, 
                             periods=PrecipitationModel.shape[0], freq='M')
        PrecipitationModel['time'] = Time
        PrecipitationModel = PrecipitationModel.sel(time=slice('1979', '2005'))
        
        print('Computing Long term climatologies')
        LongTermClim_Mean_MOD = PrecipitationModel.groupby(
            'time.season').mean('time')
        
        print('Slicing spatial')
        LongTermClim_Mean_MOD = SlicenetCDFXarray(LongTermClim_Mean_MOD,
                                                   RegionToPlot)
        
        print('Plotting data')
        SeasonalPlot(LongTermClim_Mean_MOD, axes, i, ModelNames[i])
    print('Saving figure')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(OutDirectoryFigures+'SeasonalPanels.pdf',
                format='pdf',
                dpi=30, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()
