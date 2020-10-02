import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt

def SpatialAverage(XarrayDataset,LatidudeVector):
    # Latitude vector
    latr = np.deg2rad(LatidudeVector)
    LatitudeWeights = np.array(np.cos(latr))  # Compute latitude weights
    # Calculate zonal average
    ZonalAverage = np.nanmean(np.array(XarrayDataset), axis=2)
    # Create matrix to storage nino34 index
    Nino34Index = np.zeros(XarrayDataset.shape[0])
    for j in range(XarrayDataset.shape[0]):
        # Compute spatial average over region
        Nino34Index[j] = np.average(ZonalAverage[j, :], weights=LatitudeWeights)
    return Nino34Index

def ClassifyENSOEvents(ONI,ModelId):
    ENSOClassification = np.zeros(ONI.shape,dtype='|U10')
    ONIPositive = np.array(ONI[ModelId][:] >= 0.5)
    ONINegative = np.array(ONI[ModelId][:] <= -0.5)
    #Counter
    i = 0
    while i < ONI.shape[0]:
        CountEventNino = 0
        CountEventNina = 0
        #Check for el nino
        while ONIPositive[i] == True:
            CountEventNino += 1
            i += 1
            if i == ONI.shape[0]:
                break
        #Fill for el nino
        if CountEventNino >= 5:
            ENSOClassification[i-CountEventNino:i] = 'Nino'
        if i == ONI.shape[0]:
                break
        #Check for la nina
        while ONINegative[i] == True:
            CountEventNina += 1
            i += 1
            if i == ONI.shape[0]:
                break
        #Fill for la Nina
        if CountEventNina >= 5:
            ENSOClassification[i-CountEventNina:i] = 'Nina'
        i += 1
        if i == ONI.shape[0]:
            break
    #Fill neutral
    ENSOClassification[np.where(ENSOClassification == '')] = 'Neutral'
    #Convert into a Dataframe
    ENSOClassification = pd.DataFrame(
        ENSOClassification, columns=['Event'])
    return ENSOClassification

def ComputeOni(ModelNameDirection, varname,*argv):
    #for ModelNameDirection in FileDirections[0:2]:
    netCDF4File = xr.open_dataset(ModelNameDirection)
    #Activate only if necessary
    # netCDF4File = xr.open_dataset(ModelNameDirection, decode_times=False)
    # time = pd.date_range(start='1850',periods=netCDF4File[varname].shape[0],freq='M')
    # netCDF4File['time'] = time
    VariableName = varname #list(netCDF4File.data_vars)[-1]
    #Select the correct names
    LongitudeName = 'lon'  # list(netCDF4File.coords)[0]
    LatitudeName = 'lat'  # list(netCDF4File.coords)[1]
    TimeName = 'time'  # list(netCDF4File.coords)[2]
    while True:
        try:
            #Check Longitude boundaries
            if np.amin(netCDF4File[VariableName][LongitudeName]) >= 0:
                LongitudeConversionFactor = 360
            else:
                LongitudeConversionFactor = 0
            break
        except:
            print('Wrong dimension names')
            print(list(netCDF4File.coords))
            LongitudeName = str(input('Input correct name for Longitude dimension: '))
            LatitudeName = str(input('Input correct name for Latitude dimension: '))
    #Cut files to El Nimo 3.4 region
    Nino34Data = netCDF4File[VariableName].where(
        (LongitudeConversionFactor - 170 <= netCDF4File[VariableName][LongitudeName]) &
            (LongitudeConversionFactor - 120 >= netCDF4File[VariableName][LongitudeName]) &
            (-5 < netCDF4File[VariableName][LatitudeName]) &
            (5 > netCDF4File[VariableName][LatitudeName]), drop=True)
    #Convert from Kelvin to °C
    Nino34Data = Nino34Data - 273.15
    #Slice netcdf    
    BeginBasePeriod = argv[0]
    EndBasePeriod = argv[1]
    BasePeriod = Nino34Data.sel(
        time=slice(BeginBasePeriod, EndBasePeriod))
    MonthlyClimatologies = BasePeriod.groupby('time.month').mean(dim='time')
    #Compute Anomalies
    SSTAnomalies = BasePeriod.groupby('time.month') - MonthlyClimatologies
    #Compute spatial average
    LatitudeVector = SSTAnomalies[LatitudeName] #This gives a two dimensiona array
    
    if len(LatitudeVector.shape) > 1:
        Nino34Index = SpatialAverage(SSTAnomalies, LatitudeVector[:,0])
    else:
        Nino34Index = SpatialAverage(SSTAnomalies, LatitudeVector)
    #Conver Array to a Pandas DataFrame
    ColumnName = netCDF4File.attrs['model_id']
    #ColumnName = ModelNameDirection.split('/')[-1].replace('.nc', '')
    Nino34IndexDataframe = pd.DataFrame(Nino34Index, columns=[ColumnName],
                                        index=SSTAnomalies[TimeName].to_dataframe().index)
    #Compute and classify ENSO events
    ONI = Nino34IndexDataframe.rolling(window=3,min_periods=1,center=True).mean()
    ENSOClassification = ClassifyENSOEvents(ONI, ColumnName)
    ENSOClassification.index = SSTAnomalies[TimeName].to_dataframe().index
    ONI = pd.concat([ONI, ENSOClassification], axis=1)
    return ONI


if __name__ == '__main__':
    #Files directory
    InDirectory = '/path/directory/where/the/nc_files/are/stored/'
    OutDirectory = '/path/dictory/where/results/will/be/saved/'
    #Sort files
    FileDirections = np.sort(np.array(glob.glob(InDirectory + "*.nc")))
    Start = 'start date'
    End = 'end date'
    for ModelNameDirection in FileDirections:
        print('Computing %s' % ModelNameDirection)
        ONI = ComputeOni(ModelNameDirection, 'tos', Start, End)
        ONI.to_csv(OutDirectory + ONI.columns[0] + '_' + Start + '-' + End + '.csv')
