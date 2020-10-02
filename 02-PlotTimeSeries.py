import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pylab import cm
import string


def AppendONIS(DirPaths, ModelName):
    df = pd.DataFrame()
    for i, Path in enumerate(DirPaths):
        #Open csv
        window = Path.split('/')[-2]
        serie = pd.read_csv(Path+ModelName+'_' + window + '.csv', index_col=0)
        #append
        df = pd.concat([df, serie], axis=0)
    #print(df.shape)
    TimeIndex = pd.date_range(
        start='2006', periods=df.shape[0], freq='M')
    return df


def PlotTimeSeries(TimeSeries, ModelName, axes, *args):
    #Convert index to time series object
    TimeSeries.index = pd.to_datetime(TimeSeries.index)
    #Plot time series
    axes[args[0]].plot(TimeSeries[ModelName], color='k')
    #Fill shades with colors
    axes[args[0]].fill_between(TimeSeries.index, TimeSeries[ModelName], -0.5,
                               where=TimeSeries['Event'] == 'Nina', color='blue')
    axes[args[0]].fill_between(TimeSeries.index, TimeSeries[ModelName], 0.5,
                               where=TimeSeries['Event'] == 'Nino', color='red')
    #Dotted horizontal line
    axes[args[0]].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    axes[args[0]].axhline(y=-0.5, color='black', linestyle='--', linewidth=1)
    #Format and label axis
    axes[args[0]].set_ylabel('ONI ($^{\circ}$C)')
    axes[args[0]].set_xlabel('Years')

    
    pass


def main():
    ONIHist = '/path/directory/oni/files/historical/'
    ONIFiles = '/path/directory/oni/files/cmip/'
    OutDirectory = '/path/dictory/where/results/will/be/saved/'
    #List oni files
    ONIDir = np.sort(np.array(glob.glob(ONIFiles+ "*.csv")))
    #Setup figure canvas
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5),
                             gridspec_kw={'wspace': 0, 'hspace': 0},
                             sharex=True, sharey=True)
    #List model names
    #This depends strongly on the models you're using. Change if necessary
    ModelNames = ['GFDL-ESM2M', 'GISS-E2-H', 'MIROC5']
    
    #Open files in a loop
    for i in range(len(ModelNames)):
        #Open ONI file
        ONI = pd.read_csv(ONIDir[i], sep=',', index_col=0)
        PlotTimeSeries(ONI, ModelNames[i], axes, i)
        # print(ONI)
    for i in range(axes.shape[0]):
        axes[i].text(0.01, 0.1,
                     #  string.ascii_uppercase[p],
                     ModelNames[i],
                     transform=axes[i].transAxes,
                     size=12)
    
    fig.savefig(OutDirectory+'ONITimeSeries.pdf',
                format='pdf',
                dpi=30, bbox_inches='tight')
    plt.show()

    return False


if __name__ == '__main__':
    main()
