"""
# ==================================================
This script takes processed Raman spectra and fitted peak data of plastic samples and generates analysis figures

Dependencies:
    1) Raman_processing script (to process raw spectra and output baselined spectra as .csv files)
    2) Raman_fitting script (to fit peaks in baselined spectra and output fitted peak parameters as .csv files)

Data import relies on spectra files using the following naming convention:
    MeasurementDate_SampleID_LaserWavelength_LaserPower_Accumulations-x-ExposureTime_SpectralRange_PointsAveraged_spectra.csv
This can be amended by changing lines 483-502

Data import relies on peak fit files using the following naming convention:
    MeasurementDate_SampleID_LaserWavelength_LaserPower_Accumulations-x-ExposureTime_SpectralRange_PointsAveraged_parameters.csv
This can be amended by changing lines 483-502

# ==================================================
"""

# ==================================================
# import necessary Python packages

import os
import math
import glob
import datetime
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==================================================
# user defined variables

# import spectra taken with which wavelength?
Laser_Wavelength = '*'              # wavelength as integer, or '*' for all available files

# directory of folder containing spectra files to be imported and processed
Data_dir = './output/'

# directory of sample info database containing list of sample names and their material assignments
Database_dir = './data/Database.csv'

# directory of folder where figures will be saved
Fig_dir = './figures/'

# directory of folder where processed data will be saved
Out_dir = './output/'

# specify which figures to generate
Plot_sample_summary = True      # produce a figure for each unique sample
Plot_material_summary = True    # produce a figure for each unique material
Plot_fit_summary = True         # produce figures for peak fitting results

# specify which data analysis processes to run
Do_PCA = True                   # run Principal Component Analysis on spectra
if Do_PCA == True:
    Do_clustering = True        # run K-means clustering on PCA coordinates

# parameters for selecting which spectra to analyse
Target_params = {
    '532': {
        'laser_power': 1,
        'accumulations': 20,
        'exposure_time': 0.5,
        'regions': [(400, 1800), (2800, 3200)]
    },
    '638': {
        'laser_power': 10,
        'accumulations': 10,
        'exposure_time': 1,
        'regions': [(400, 1800), (2800, 3200)]
    },
    '785': {
        'laser_power': 100,
        'accumulations': 10,
        'exposure_time': 10,
        'regions': [(400, 1800), (2800, 3200)]
    }
}

# list expected materials and their plotting colours
Materials = ['not plastic', 'pumice', 'unassigned', 'PE', 'LDPE', 'MDPE', 'HDPE', 'PP', 'PPE', 'PS', 'PLA', 'ABS', 'EAA', 'PA']
Material_colors =  ['k', 'skyblue', 'chartreuse', 'r', 'g', 'gold', 'b', 'm', 'y', 'tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'seagreen']

# generic lists of colours and markers for plotting
Color_list =  ['r', 'b', 'm', 'y', 'tab:gray', 'c', 'tab:orange', 'tab:brown', 'tab:pink', 'tan']
Marker_list = ['o', 's', 'v', '^', 'D', '*', 'p']

# laser power in mW, measured at sample
Laser_power = {
    '532': 0.44,
    '638': 1.56,
    '785': 21.4
}

# list of all laser wavelengths being considered (for consistent plotting)
All_Lasers = ['248', '532', '633', '638', '785', '830', '1064']
Laser_colors = ['tab:purple', 'g', 'orangered', 'r', 'firebrick', 'tab:brown', 'k']

# dict containing key,item pairs for materials and lists of their known peak positions (for plotting)
Peak_positions = {
    'PE': [1060, 1125, 1295, 1440, 1460, 2845, 2880],
    'PP': [810, 840, 975, 1150, 1170, 1330, 1455, 2840, 2860, 2875, 2900, 2920, 2950],
    'PPE': [810, 840, 975, 1150, 1170, 1330, 1455, 2840, 2860, 2875, 2900, 2920, 2950],
    'PS': [995, 1030, 1600, 2850, 2900, 3050]
}

"""
# ==================================================
# import sample info database
# ==================================================
"""

print()
print()
print("importing sample database...")

# import database and remove NaNs
sample_database = pd.read_csv(Database_dir, header=0, true_values="Y", false_values="N")
sample_database.fillna(value="", inplace=True)

print()
print(sample_database.info())

Sample_IDs = np.unique(sample_database['ID'])

"""
# ==================================================
# define functions for handling databases
# ==================================================
"""

def get_chemicalID(dataframe, sample_ID, debug=False):
    # get sample material assignment from dataframe
    yesno = dataframe['ID'] == sample_ID
    sliced_dataframe = dataframe[yesno]
    if debug == True:
        print("sliced rows:", len(sliced_dataframe))
    assignment = ""
    if len(sliced_dataframe) > 0:
        temp = sliced_dataframe['Assignment'].iloc[0]
        if debug == True:
            print("material assignment:", temp, type(temp))
        if temp != np.nan:
            assignment = temp
    return assignment

"""
# ==================================================
# define functions for handling Raman spectra
# ==================================================
"""

# ==================================================
# functions for selecting suitable spectra

def get_best_sample_spectrum(datadict, sample, laser, goal='points', strict=True, x_start=None, x_end=None, debug=False):
    # function for finding the most appropriate spectrum for a given sample
    global Target_params
    if debug == True:
        print("getting best spectrum for %s..." % sample)
    if x_start != None and x_end != None:
        # filter for spectra that match sample, laser, and contain the range x_start to x_end
        specsort = np.ravel(np.where((datadict['sample'] == sample) & (datadict['laser_wavelength'] == laser) & (x_start >= datadict['x_start']) & (datadict['x_end'] >= x_end)))
        if debug == True:
            print("    required range: %0.f-%0.f cm" % (x_start, x_end))
            print("    %s spectra found that match sample, laser, and spectral range..." % len(specsort))
    else:
        # filter for spectra that match sample and laser only
        specsort = np.ravel(np.where((datadict['sample'] == sample) & (datadict['laser_wavelength'] == laser)))
        if debug == True:
            print("    %s spectra found that match sample and laser..." % len(specsort))
    
    # if at least one spectrum is found that matches filters, proceed
    if len(specsort) > 0:
        if goal.lower() in ['target_params', 'target params', 'target'] and laser != None:
            # use wavelength-specific target parameters defined by Target_params
            params = Target_params[str(laser)]
            sort = specsort[np.ravel((datadict['laser_power'][specsort] == params['laser_power']) & (datadict['accumulations'][specsort] == params['accumulations']) & (datadict['exposure_time'][specsort] == params['exposure_time']))]
            if debug == True:
                print("    getting best spectrum based on measurement settings...")
                print("        target params: %3.1f%% %2.fx%2.1fs %0.f-%0.f cm" % (params['laser_power'], params['accumulations'], params['exposure_time'], x_start, x_end))
            if len(sort) > 0:
                best_index = sort[0]
                if debug == True:
                    print("        data settings:")
                    for i in sort:
                        print("            %4s: %3.1f%%, %2.fx%2.1fs, %0.f-%0.f cm" % (i, datadict['laser_power'][i], datadict['accumulations'][i], datadict['exposure_time'][i], datadict['x_start'][i], datadict['x_end'][i]))
                    print("        best params %4s: %3.1f%%, %2.fx%2.1fs, %0.f-%0.f cm" % (best_index, datadict['laser_power'][best_index], datadict['accumulations'][best_index], datadict['exposure_time'][best_index], datadict['x_start'][i], datadict['x_end'][i]))
                return best_index
            else:
                return None
        elif goal.lower() in ['fitted_peaks', 'peaks']:
            # prioritise spectra with fitted peak data, then points, then energy
            sort = specsort[np.lexsort((datadict['total_energy'][specsort], datadict['points'][specsort], datadict['fit_yesno'][specsort]))]
            if debug == True:
                print("    getting spectrum with best SNR and fitted peaks...")
            if len(sort) > 0:
                best_index = sort[-1]
                if debug == True:
                    print("        data with fits, by points and exposure:")
                    for i in sort:
                        print("            %4s: fit=%5s, %2.f points, %6.1f exposure" % (i, datadict['fit_yesno'][i], datadict['points'][i], datadict['total_energy'][i]))
                    print("        best range %4s: fit=%5s, %2.f points, %6.1f exposure" % (i, datadict['fit_yesno'][best_index], datadict['points'][best_index], datadict['total_energy'][best_index]))
                return best_index
            else:
                return None
        elif goal.lower() in ['largest_range', 'largest range', 'longest_range', 'longest range', 'range']:
            # prioritise range, then points, then energy
            sort = specsort[np.lexsort((datadict['total_energy'][specsort], datadict['points'][specsort], datadict['x_range'][specsort]))]
            if debug == True:
                print("    getting spectrum with largest spectral range....")
            if len(sort) > 0:
                if debug == True:
                    print("        data by range:")
                    for i in sort:
                        print("            %4s: %4.f cm, %2.f points, %6.f exposure" % (i, datadict['x_range'][i], datadict['points'][i], datadict['total_energy'][i]))
                    print("            %4s: %4.f cm, %2.f points, %6.f exposure" % (best_index, datadict['x_range'][best_index], datadict['points'][best_index], datadict['total_energy'][best_index]))
                return best_index
            else:
                return None
        elif goal.lower() in ['highest_energy', 'highest energy', 'high_energy', 'high energy', 'energy']:
            # prioritise total energy, then points, then range
            sort = specsort[np.lexsort((datadict['x_range'][specsort], datadict['points'][specsort], datadict['total_energy'][specsort]))]
            if debug == True:
                print("    getting spectrum with highest exposure energy....")
            if len(sort) > 0:
                best_index = sort[-1]
                if debug == True:
                    print("        data by exposure:")
                    for i in sort:
                        print("            %4s: %6.f exposure, %2.f points, %4.f cm" % (i, datadict['total_energy'][i], datadict['points'][i], datadict['x_range'][i]))
                    print("            %4s: %6.f cm, %2.f points, %4.f exposure" % (best_index, datadict['x_range'][best_index], datadict['points'][best_index], datadict['total_energy'][best_index]))
                return best_index
            else:
                return None
        else:
            # prioritise number of points, then energy, then range
            sort = specsort[np.lexsort((datadict['x_range'][specsort], datadict['total_energy'][specsort], datadict['points'][specsort]))]
            if debug == True:
                print("    getting spectrum averaged from most points....")
            if len(sort) > 0:
                best_index = sort[-1]
                if debug == True:
                    print("        data by exposure:")
                    for i in sort:
                        print("            %4s: %2.f points, %6.f exposure, %4.f cm" % (i, datadict['points'][i], datadict['total_energy'][i], datadict['x_range'][i]))
                    print("            %4s: %2.f points, %6.f exposure, %4.f cm" % (best_index, datadict['points'][best_index], datadict['total_energy'][best_index], datadict['x_range'][best_index]))
                return best_index
            else:
                return None
    else:
        return None

# ==================================================
# functions for converting between raman shift, wavelength, and photon energy

def wavelength2shift(wavelength, excitation=785):
    if type(excitation) in [str, int]:
        excitation = float(excitation)
    shift = ((1./excitation) - (1./wavelength)) * (10**7)
    return shift

def shift2wavelength(shift, excitation=785):
    if type(excitation) in [str, int]:
        excitation = float(excitation)
    wavelength = 1./((1./excitation) - shift/(10**7))
    return wavelength

def wavelength2energy(wavelength):
    # convert photon wavelength (in nm) to energy (in J)
    hc = 1.98644586*10**-25 # in J meters
    return hc / (np.asarray(wavelength)/10**9)

def energy2wavelength(energy):
    # convert photon energy (in J) to wavelength (in nm)
    hc = 1.98644586*10**-25 # in J meters
    return (hc / np.asarray(wavelength)) * 10**9

def photon_count(wavelength, energy):
    # convert a total laser energy (in J) to a photon count
    hc = 1.98644586*10**-25 # in J meters
    photon_energy = hc / (np.asarray(wavelength)/10**9)
    return np.asarray(energy) / photon_energy

# ==================================================
# functions for converting Y axis values

def intensity2snr(intensity, noise):
    return intensity / noise

def snr2intensity(snr, noise):
    return snr * noise

# ==================================================
# functions for finding and fitting peaks

def find_max(x, y, x_start, x_end):
    # function for finding the maximum in a slice of input data
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]       # create slice
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmax(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the maximum

def find_min(x, y, x_start, x_end):
    # function for finding the minimum in a slice of input data
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]       # create slice
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmin(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the minimum

def find_maxima(x, y, window_length, threshold, debug=False):
    # function for finding the maxima of input data. Each maximum will have the largest value within its window.
    index_list = argrelextrema(y, np.greater, order=window_length)  # determines indices of all maxima
    all_maxima = np.asarray([x[index_list], y[index_list]]) # creates an array of x and y values for all maxima
    y_limit = threshold * np.amax(y)                        # set the minimum threshold for defining a 'peak'
    x_maxima = all_maxima[0, all_maxima[1] >= y_limit]      # records the x values for all valid maxima
    y_maxima = all_maxima[1, all_maxima[1] >= y_limit]      # records the y values for all valid maxima
    maxima = np.asarray([x_maxima, y_maxima])               # creates an array for all valid maxima
    if debug == True:
        print("        maxima:")
        print("        x: ", maxima[0])
        print("        y: ", maxima[1])
    return maxima

"""
# ==================================================
# import spectra
# ==================================================
"""

# ==================================================
# find spectra files

print()
print("searching for spectra files...")

files = sorted(glob.glob("%s%snm/by sample/*/*_spectra.csv" % (Data_dir, str(Laser_Wavelength))))

print()
print("    %s files found" % len(files))

raman = {
    'ID': [],
    'measurement_date': [],
    'sample': [],
    'wavelength': [],
    'raman_shift': [],
    'y_av': [],
    'y_std': [],
    'y_av_norm': [],
    'y_std_norm': [],
    'y_av_sub': [],
    'y_std_sub': [],
    'y_av_sub_norm': [],
    'y_std_sub_norm': [],
    'raw_spec': [],
    'x_start': [],
    'x_end': [],
    'x_range': [],
    'laser_wavelength': [],
    'laser_power': [],
    'accumulations': [],
    'exposure_time': [],
    'total_energy': [],
    'photon_count': [],
    'material': [],
    'points': [],
    'fig_dir': [],
    'out_dir': [],
    'fit_yesno': [],
    'fitted_peaks': []
}

count = 0
for file in files:
    if 'background' not in file:
        count += 1
        filename = file.split("/")[-1]
        print()
        print("    %i:" % count, filename)
        yesno = False   # tracks whether spec import was successful
        while True:
            try:
                print("        importing %s..." % filename)
                # extract info from filename
                sample = filename.split("_")[1]
                print("        sample:", sample)
                date = datetime.datetime.strptime(filename.split("_")[0], "%Y-%m-%d")
                print("        measured on:", date.strftime("%Y-%m-%d"))
                
                # extract measurement info
                print("        spectrum properties:")
                laser = filename.split("_")[2][:-2]
                print("            laser: %s nm" % laser)
                power = float(filename.split("_")[3][:-1])
                print("            power: %0.f%%" % power)
                accum = float(filename.split("_")[4].split("x")[0])
                exposure = float(filename.split("_")[4].split("x")[1][:-1])
                print("            %s accumulations of %s seconds" % (accum, exposure))
                total_energy = float(exposure)*(float(power)/float(Laser_power[str(laser)]))
                print("        total energy used: %0.1f mJ" % total_energy)
                print("            in photons:", photon_count(float(laser), float(total_energy)))
                x_start, x_end = filename.split("_")[5][:-2].split("-")
                print("        spectral range: %s - %s cm-1" % (x_start, x_end))
                points = int(filename.split("_")[6].split("-")[0])
                print("        averaged over %i points" % points)
                
                # import data array
                spec = np.genfromtxt(file, delimiter=',')
                print("        spec array:", np.shape(spec))
                print("            nan check:", np.any(np.isnan(spec)))
                print("            inf check:", np.any(np.isinf(spec)))
                if np.size(spec, axis=1) < np.size(spec, axis=0):
                    # transpose array
                    spec = spec.transpose()
                    print("            transposed spec array:", np.shape(spec))
                x_start, x_end = (spec[1,0], spec[1,-1])
                print("            true spectral range: %0.f - %0.f cm-1" % (x_start, x_end))
                print("                in nm: %0.f - %0.f nm" % (spec[0,0], spec[0,-1]))
                x_start, x_end = filename.split("_")[5][:-2].split("-")
                print("            raw spectra found:", np.size(spec, axis=0)-9)
                if np.size(spec, axis=0)-9 != points:
                    print("                does not match reported number of points!")
                
                if np.size(spec, axis=0) >= 9:
                    # all parameters obtained, add data to arrays
                    id_temp = "%s_%snm_%.2f%%_%0.fx%.2fs" % (sample, str(laser), power, accum, exposure)
                    print("        adding data to array")
                    raman['ID'].append(id_temp)
                    raman['measurement_date'].append(date)
                    raman['sample'].append(sample)
                    raman['x_start'].append(float(x_start))
                    raman['x_end'].append(float(x_end))
                    raman['x_range'].append(float(x_end)-float(x_start))
                    raman['laser_wavelength'].append(str(laser))
                    raman['laser_power'].append(power)
                    raman['accumulations'].append(accum)
                    raman['exposure_time'].append(exposure)
                    raman['total_energy'].append(total_energy)
                    raman['photon_count'].append(photon_count(float(laser), float(total_energy)))
                    raman['points'].append(points)
                    keys = ['wavelength', 'raman_shift', 'y_av', 'y_std', 'y_av_norm', 'y_std_norm', 'y_av_sub', 'y_std_sub', 'y_av_sub_norm', 'y_std_sub_norm']
                    for i, key in zip(range(len(keys)), keys):
                        raman[key].append(spec[i])
                    raman['raw_spec'].append(spec[9:])
                        
                    # check if sample has material assignment in sample database:
                    assignment = get_chemicalID(sample_database, sample)
                    if assignment != '':
                        raman['material'].append(assignment)
                    else:
                        raman['material'].append("unassigned")
                    
                    # create folders for figures, data output
                    fig_dir = "%s%snm/by sample/%s/" % (Fig_dir, str(laser), sample)
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)
                    raman['fig_dir'].append(fig_dir)
                    out_dir = "%s%snm/by sample/%s/" % (Out_dir, str(laser), sample)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    raman['out_dir'].append(out_dir)
                    
                    yesno = True
                    print("        import finished!")
                    break
                else:
                    print("        not enough data in array, %s columns missing!" % (11-np.size(spec, axis=0)))
                    break
            except Exception as e:
                print("        something went wrong!")
                print("            type error:", str(e))
                break
        # then attempt to import peak fit parameters
        if yesno == True:
            # create temporary datadict for peak parameters
            fitted_peaks = {'center': [], 'amplitude': [], 'fwhm': [], 'center standard error': [], 'amplitude standard error': [], 'fwhm standard error': []}
            # boolean to track whether fit data successfully imported
            fit_yesno = False
            folder = "/".join(file.split("/")[:-1])+"/"
            while True:
                try:
                    print("        now importing fitting parameters for %s..." % id_temp)
                    fit_filename = "%s_%s_%0.f-%0.fcm_*-fit-parameters.csv" % (date.strftime("%Y-%m-%d"), id_temp, float(x_start), float(x_end))
                    print("            ", fit_filename)
                    print("%s%s" % (folder, fit_filename))
                    fit_dirs = glob.glob("%s%s" % (folder, fit_filename))
                    if len(fit_dirs) > 0:
                        # range check
                        fit_start, fit_end = fit_dirs[0].split("_")[6][:-2].split("-")
                        if np.abs(float(x_start) - float(fit_start)) > 1.:
                            print("             fit x_start >1 cm-1 from spec x_start")
                        if np.abs(float(x_end) - float(x_end)) > 1.:
                            print("             fit x_end >1 cm-1 from spec x_end")
                        # set up arrays depending on specified function
                        function = fit_dirs[0].split("_")[-1].split("-")[0]
                        print("        fitting function used:", function)
                        if function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
                            function = 'pv'
                            fitted_peaks['sigma'] = []
                            fitted_peaks['sigma standard error'] = []
                            fitted_peaks['eta'] = []
                            fitted_peaks['eta standard error'] = []
                        elif function.lower() in ['l', 'lorentz', 'lorentzian']:
                            function = 'l'
                            fitted_peaks['gamma'] = []
                            fitted_peaks['gamma standard error'] = []
                        else:
                            function = 'g'
                            fitted_peaks['sigma'] = []
                            fitted_peaks['sigma standard error'] = []
                        # import fit data as dataframe
                        fit_data = pd.read_csv(fit_dirs[0], delimiter=', ')
                        print("        fit data array:", np.shape(fit_data))
                        print("            columns:", fit_data.columns)
                        fit_data.rename(columns={'# center':'center'}, inplace=True)
                        # add data to fitted_peaks dict
                        for key in fit_data.columns:
                            fitted_peaks[key] = np.asarray(fit_data[key])
                        fit_yesno = True
                        print("        fitting import finished!")
                        break
                    else:
                        print("        no fit parameters file found")
                        break
                except Exception as e:
                    print("        something went wrong!")
                    print("            type error:", str(e))
                    break
            # add fit data to array
            raman['fit_yesno'].append(fit_yesno)
            # convert results to numpy arrays and add to storage array
            raman['fitted_peaks'].append({})
            for key in fitted_peaks.keys():
                raman['fitted_peaks'][-1][key] = np.asarray(fitted_peaks[key])
                
print()

# clean up arrays
for key in raman.keys():
    if key not in ['y']:
        raman[key] = np.asarray(raman[key])
        if len(raman[key]) != len(raman['ID']):
            print("    WARNING: %s array length does not match ID array length!" % key)

# trim iterable lists down to those with FTIR data only
Sample_IDs = [sample for sample in Sample_IDs if sample in raman['sample']]
materials = [material for material in Materials if np.any(sample_database['Assignment'] == material)]
Lasers = [str(item) for item in All_Lasers if str(item) in np.unique(raman['laser_wavelength'])]

print()
print("lasers found in Raman data: ", ", ".join(["%s nm" % laser for laser in Lasers]))
print("samples found in Raman data:", len(Sample_IDs))
print("spectra found in Raman data:", len(raman['ID']))

# report how many spectra found for each sample at each wavelength, by date
print()
for laser in Lasers:
    result = np.ravel(np.where(raman['laser_wavelength'] == laser))
    print("%s nm: %s spectra imported" % (laser, len(result)))
    
# generate material-specific figure and output folders
spec_count = 0
print()
for material in materials:
    result = np.ravel(np.where(raman['material'] == material))
    if len(result) > 0:
        print("%s: %s spectra total" % (material, len(result)))
        for laser in Lasers:
            result = np.ravel(np.where((raman['material'] == material) & (raman['laser_wavelength'] == laser)))
            if len(result) > 0:
                print("    %s nm: %d spectra" % (laser, len(result)))
                spec_count += len(result)
                figdir = '%s%snm/by material/%s/' % (Fig_dir, laser, material)
                if not os.path.exists(figdir):
                    os.makedirs(figdir)
                outdir = '%s%snm/by material/%s/' % (Out_dir, laser, material)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                figdir = '%sby material/%s/' % (Fig_dir, material)
                if not os.path.exists(figdir):
                    os.makedirs(figdir)
print("total spectra with material assignments:", spec_count)

# generate colourmap for total energy used to acquire spectrum
cmin = 0.75*np.amin(raman['total_energy'])
cmax = 1.33*np.amax(raman['total_energy'])
norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])

"""
# ==================================================
# collate data by wavelength and generate figures
# ==================================================
"""

print()
print("collating data by sample...")

normalise = True
x_start = 400
x_end = 4000

sample_data = {}

show_plots = False

for laser in Lasers:
    # create datadict for this laser wavelength
    sample_data[laser] = {'x': np.linspace(x_start, x_end, 2*int(x_end-x_start))+1, 'y_full': [], 'y_full_norm': [], 'y_fingerprint': [], 'y_fingerprint_norm': [], 'material': [], 'sample': []}
    
    for sample in Sample_IDs:
        # collate best available spectra for each sample
        material = get_chemicalID(sample_database, sample)
        
        # get best full-range spectrum
        full_index = get_best_sample_spectrum(raman, sample, laser=laser, goal='highest_energy', x_start=600, x_end=3300, debug=False)
        
        # get best fingerprint-range spectrum
        fingerprint_index = get_best_sample_spectrum(raman, sample, laser=laser, goal='highest_energy', x_start=600, x_end=1800, debug=False)
        
        if full_index != None and fingerprint_index != None:
            # at least one spectrum found, add data to arrays
            sample_data[laser]['material'].append(material)
            sample_data[laser]['sample'].append(sample)
            y_full = np.interp(sample_data[laser]['x'], raman['raman_shift'][full_index], raman['y_av_sub'][full_index], left=np.nan, right=np.nan)
            y_full_norm = np.interp(sample_data[laser]['x'], raman['raman_shift'][full_index], raman['y_av_sub_norm'][full_index], left=np.nan, right=np.nan)
            sample_data[laser]['y_full'].append(y_full)
            sample_data[laser]['y_full_norm'].append(y_full_norm)
            y = np.interp(sample_data[laser]['x'], raman['raman_shift'][fingerprint_index], raman['y_av_sub'][fingerprint_index], left=np.nan, right=np.nan)
            y_norm = np.interp(sample_data[laser]['x'], raman['raman_shift'][fingerprint_index], raman['y_av_sub_norm'][fingerprint_index], left=np.nan, right=np.nan)
            sample_data[laser]['y_fingerprint'].append(y)
            sample_data[laser]['y_fingerprint_norm'].append(y_norm)
            if Plot_sample_summary == True:
                # plot results
                plt.figure(figsize=(8,8))
                plt.subplot(211)
                plt.title("%s (%s) %s nm Raman Spectra\n%3.f%% %2.fx%0.1fs %0.f-%0.f cm-1" % (sample, material, laser, raman['laser_power'][full_index], raman['accumulations'][full_index], raman['exposure_time'][full_index], raman['x_start'][full_index], raman['x_end'][full_index]))
                color = Material_colors[Materials.index(material)]
                y_max = 0.
                for i in range(raman['points'][full_index]):
                    plt.plot(raman['raman_shift'][full_index], raman['raw_spec'][full_index][i], color, label='spec %s' % (i+1))
                    if np.amax(raman['raw_spec'][full_index][i]) > y_max:
                        y_max = np.amax(raman['raw_spec'][full_index][i])
                if raman['points'][full_index] > 1:
                    plt.plot(raman['raman_shift'][full_index], np.mean(raman['raw_spec'][full_index], axis=0), 'k', label='%s-pt average' % raman['points'][full_index])
                plt.ylim(-0.2*y_max, 1.2*y_max)
                plt.ylabel("Raman Intensity")
                plt.xlim(x_start, x_end)
                plt.legend()
                plt.subplot(212)
                plt.title("%3.f%% %2.fx%0.1fs %0.f-%0.f cm-1" % (raman['laser_power'][fingerprint_index], raman['accumulations'][fingerprint_index], raman['exposure_time'][fingerprint_index], raman['x_start'][fingerprint_index], raman['x_end'][fingerprint_index]))
                y_max = 0.
                for i in range(raman['points'][fingerprint_index]):
                    plt.plot(raman['raman_shift'][fingerprint_index], raman['raw_spec'][fingerprint_index][i], color, label='spec %s' % (i+1))
                    if np.amax(raman['raw_spec'][fingerprint_index][i]) > y_max:
                        y_max = np.amax(raman['raw_spec'][fingerprint_index][i])
                if raman['points'][fingerprint_index] > 1:
                    plt.plot(raman['raman_shift'][fingerprint_index], np.mean(raman['raw_spec'][fingerprint_index], axis=0), 'k', label='%s-pt average' % raman['points'][fingerprint_index])
                plt.ylim(-0.2*y_max, 1.2*y_max)
                plt.ylabel("Raman Intensity")
                plt.xlim(x_start, x_end)
                plt.xlabel("Raman Shift (cm$^{-1}$)")
                plt.legend()
                plt.minorticks_on()
                plt.tight_layout()
                plt.savefig("%s/%s_summary.png" % (raman['fig_dir'][full_index], raman['ID'][full_index]), dpi=300)
                if show_plots == True:
                    plt.show()
                else:
                    plt.close()
                # plot spectral standardisation (using fingerprint spec)
                sliced = np.ravel(np.where((400 <= sample_data[laser]['x']) & (sample_data[laser]['x'] <= 1800)))
                y_snv = (y - np.mean(y[sliced])) / np.std(y[sliced])
                y_snv_1 = savgol_filter(y, 25, polyorder = 5, deriv=1)
                plt.figure(figsize=(8,4))
                plt.subplot(211)
                plt.title("%s\nStandardised" % (raman['ID'][fingerprint_index]))
                plt.plot(sample_data[laser]['x'][sliced], y_snv[sliced], 'k')
                plt.xlim(x_start, x_end)
                plt.minorticks_on()
                plt.subplot(212)
                plt.title("First Derivative")
                plt.plot(sample_data[laser]['x'][sliced], y_snv_1[sliced], 'k')
                plt.xlim(x_start, x_end)
                plt.xlabel("Raman Shift (cm^{-1})")
                plt.minorticks_on()
                plt.tight_layout()
                plt.savefig("%s/%s_standardised.png" % (raman['fig_dir'][fingerprint_index], raman['ID'][fingerprint_index]), dpi=300)
                if show_plots == True:
                    plt.show()
                else:
                    plt.close()
            
    print()
    print("%snm:" % laser)
    for key in sample_data[laser].keys():
        sample_data[laser][key] = np.asarray(sample_data[laser][key])
        print("    %s" % key, np.shape(sample_data[laser][key]))

"""
# ==================================================
# plot spectra by material
# ==================================================
"""

offset = 0.
normalise = True
norm_each_region = True

if Plot_material_summary == True:
    for laser in Lasers:
        for material in Materials:
            color = Material_colors[Materials.index(material) % len(Material_colors)]
            result = np.ravel(np.where(sample_data[laser]['material'] == material))
            if len(result) > 0:
                print()
                print("plotting %s spectra for %s samples" % (len(result), material))
                plt.figure(figsize=(10,8))
                ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
                ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
                ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
                ax1.set_title("%s spectra, %snm" % (material, laser))
                ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax1.set_xlim(400, 4000)
                ax2.set_xlim(400, 1800)
                ax3.set_xlim(2800, 3100)
                if normalise == True:
                    ax1.set_ylabel("Normalised Intensity")
                    ax2.set_ylabel("Normalised Intensity")
                    ax1.set_ylim(0, 1)
                else:
                    ax1.set_ylabel("Intensity (counts)")
                    ax2.set_ylabel("Intensity (counts)")
                count = 0
                for i in result:
                    # get info for this sample
                    label = sample_data[laser]['sample'][i]
                    # get full spectrum
                    x = sample_data[laser]['x']
                    mask = np.isnan(sample_data[laser]['y_full'][i])
                    x_full = x[~mask]
                    y_full = sample_data[laser]['y_full'][i][~mask]
                    # get fingerprint spectrum
                    mask = np.isnan(sample_data[laser]['y_fingerprint'][i])
                    x_finger = x[~mask]
                    y_finger = sample_data[laser]['y_fingerprint'][i][~mask]
                    if normalise == True:
                        ax1_min = find_min(x_full, y_full, ax1.get_xlim()[0], ax1.get_xlim()[1])[1]
                        ax1_max = find_max(x_full, y_full, ax1.get_xlim()[0], ax1.get_xlim()[1])[1]
                        if norm_each_region == True:
                            ax2_min = find_min(x_finger, y_finger, ax2.get_xlim()[0], ax2.get_xlim()[1])[1]
                            ax2_max = find_max(x_finger, y_finger, ax2.get_xlim()[0], ax2.get_xlim()[1])[1]
                            ax3_min = find_min(x_full, y_full, ax3.get_xlim()[0], ax3.get_xlim()[1])[1]
                            ax3_max = find_max(x_full, y_full, ax3.get_xlim()[0], ax3.get_xlim()[1])[1]
                        else:
                            ax2_min = ax1_min
                            ax2_max = ax1_max
                            ax3_min = ax1_min
                            ax3_max = ax1_max
                        ax1.plot(x_full, y_full/ax1_max + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                        ax2.plot(x_finger, y_finger/ax2_max + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                        ax3.plot(x_full, y_full/ax3_max + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    else:
                        ax1.plot(x_full, y_full + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                        ax2.plot(x_finger, y_finger + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                        ax3.plot(x_full, y_full + count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    count += 1
                if offset > 0:
                    ax1.set_yticks([])
                    if normalise == True:
                        ax1.set_ylim(-0.2, (1.2-offset)+count*offset)
                        ax3.set_ylim(-0.2, (1.2-offset)+count*offset)
                        ax3.set_yticks([])
                else:
                    if normalise == True:
                        ax1.set_ylim(-0.2, 1.2)
                        ax3.set_ylim(-0.2, 1.2)
                        ax3.set_yticks([])
                plt.tight_layout()
                plt.savefig("%s%snm/by material/%s_spectra.png" % (Fig_dir, laser, material), dpi=300)
                plt.show()

offset = 0.8
normalise = True
norm_each_region = True

if Plot_material_summary == True:
    print()
    print("plotting mean spectra for each material")
    print(Materials)
    for laser in Lasers:
        print()
        print(laser)
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
        ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
        ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
        ax1.set_title("Mean %snm Raman Spectra by Material" % laser)
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_xlim(400, 4000)
        ax2.set_xlim(400, 1800)
        ax3.set_xlim(2800, 3100)
        if normalise == True:
            ax1.set_ylabel("Normalised Intensity")
            ax2.set_ylabel("Normalised Intensity")
            ax1.set_ylim(0, 1)
        else:
            ax1.set_ylabel("Intensity (counts)")
            ax2.set_ylabel("Intensity (counts)")
        count = 0
        for material in Materials:
            result = np.ravel(np.where(sample_data[laser]['material'] == material))
            print(material, len(result))
            color = Material_colors[Materials.index(material) % len(Material_colors)]
            if len(result) > 0:
                # get full spectra and average, ignoring nans
                x_av = sample_data[laser]['x']
                mask = np.any(np.isnan(sample_data[laser]['y_full'][result]), axis=0)
                x_full = x_av[~mask]
                y_full = np.nanmean(sample_data[laser]['y_full'][result][:,~mask], axis=0)
                # get fingerprint spectra and average, ignoring nans
                mask = np.any(np.isnan(sample_data[laser]['y_fingerprint'][result]), axis=0)
                x_finger = x_av[~mask]
                y_finger = np.nanmean(sample_data[laser]['y_fingerprint'][result][:,~mask], axis=0)
                print(np.shape(x_av), np.shape(y_full), np.shape(y_finger))
                if normalise == True:
                    ax1_min = find_min(x_full, y_full, ax1.get_xlim()[0], ax1.get_xlim()[1])[1]
                    ax1_max = find_max(x_full, y_full, ax1.get_xlim()[0], ax1.get_xlim()[1])[1]
                    if norm_each_region == True:
                        ax2_min = find_min(x_finger, y_finger, ax2.get_xlim()[0], ax2.get_xlim()[1])[1]
                        ax2_max = find_max(x_finger, y_finger, ax2.get_xlim()[0], ax2.get_xlim()[1])[1]
                        ax3_min = find_min(x_full, y_full, ax3.get_xlim()[0], ax3.get_xlim()[1])[1]
                        ax3_max = find_max(x_full, y_full, ax3.get_xlim()[0], ax3.get_xlim()[1])[1]
                    else:
                        ax2_min = ax1_min
                        ax2_max = ax1_max
                        ax3_min = ax1_min
                        ax3_max = ax1_max
                    ax1.plot(x_full, y_full/ax1_max + count*offset, color, label=material)
                    ax2.plot(x_finger, y_finger/ax2_max + count*offset, color, label=material)
                    ax3.plot(x_full, y_full/ax3_max + count*offset, color, label=material)
                else:
                    ax1.plot(x_full, y_full + count*offset, color, label=material)
                    ax2.plot(x_finger, y_finger + count*offset, color, label=material)
                    ax3.plot(x_full, y_full + count*offset, color, label=material)
                count += 1
        if offset > 0:
            ax1.set_yticks([])
            if normalise == True:
                ax1.set_ylim(-0.2, 1.2-offset+count*offset)
                ax3.set_ylim(-0.2, 1.2-offset+count*offset)
        else:
            if normalise == True:
                ax1.set_ylim(-0.2, 1.2)
                ax3.set_ylim(-0.2, 1.2)
                ax3.set_yticks([])
        ax1.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("%s%snm/by material/material_mean_spectra.png" % (Fig_dir, laser), dpi=300)
        plt.show()
                
"""
# ==================================================
# Plot peak fit data
# ==================================================
"""
    
# plot fwhm on y axis?
plot_fwhm = False
# add errorbars to all plots?
plot_errorbars = False
        
# plot fitted peak positions for all materials, by laser
if Plot_fit_summary == True and Plot_material_summary == True:
    for laser in Lasers:
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
        ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
        ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
        ax1.set_title("%snm Fitted Peak Positions" % str(laser))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
        if plot_fwhm == True:
            ax1.set_ylabel("FWHM (cm$^{-1}$)")
            ax2.set_ylabel("FWHM (cm$^{-1}$)")
            ax3.set_ylabel("FWHM (cm$^{-1}$)")
        else:
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])
        ax1.set_xlim(400, 4000)
        ax2.set_xlim(400, 1800)
        ax3.set_xlim(2800, 3100)
        labels = []
        count = 0
        sample_count = 0
        mat_temp = []
        for material in materials:
            if material not in mat_temp:
                mat_temp.append(material)
                count += 10
            if material in Peak_positions.keys():
                ref_peaks = np.unique(Peak_positions[material])
            else:
                ref_peaks = np.unique(np.concatenate([Peak_positions[key] for key in Peak_positions.keys()]))
            colour = Material_colors[Materials.index(material) % len(Material_colors)]
            samples = [raman['sample'][i] for i in range(len(raman['sample'])) if raman['material'][i] == material]
            if len(samples) == 0:
                print("    no matching %s nm data for %s" % (laser, material))
            else:
                print("    %s %s samples have %s nm data" % (len(samples), material, laser))
                for samples in samples:
                    imax = get_best_sample_spectrum(raman, samples, laser, goal='peaks', x_start=500, x_end=3300)
                    ### print("    ", samples, imax)
                    if imax != None:
                        peaks = raman['fitted_peaks'][imax]
                        if len(peaks['center']) == 0:
                            ### print("         no peaks fitted")
                            pass
                        else:
                            ### print("        %3s peaks fitted" % len(peaks['center']))
                            alphas = peaks['amplitude'] / np.amax(peaks['amplitude'])
                            if plot_fwhm == True:
                                y = peaks['fwhm']
                                y_err = peaks['fwhm standard error']
                            else:
                                y = np.full_like(peaks['center'], -count)
                                y_err = np.zeros_like(peaks['center'])
                            ### print(np.shape(peaks['center']), np.shape(y))
                            if plot_errorbars == True:
                                for i2 in range(len(peaks['center'])):
                                    if np.amin(np.abs(peaks['center'][i2] - ref_peaks)) <= 10:
                                        mfc = colour
                                    else:
                                        mfc = 'w'
                                    ax1.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                                    ax2.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                                    ax3.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                            label = "_%s" % material
                            ### print("        %s %s: %s %s" % (imax, raman['ID'][imax], label, colour))
                            ### print("            ", peaks['center'])
                            if label not in labels:
                                labels.append(label)
                                label = "%s" % material
                            for i2 in range(len(peaks['center'])):
                                if np.amin(np.abs(peaks['center'][i2] - ref_peaks)) <= 10:
                                    mfc = colour
                                else:
                                    mfc = 'w'
                                ax1.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                                ax2.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                                ax3.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                            count += 1
                            sample_count += 1
        plt.figtext(0.15, 0.5, "%s/%s samples have fitted peaks" % (sample_count, len(np.unique(raman['sample'][np.where(raman['laser_wavelength'] == laser)]))))
        ### ax1.legend()
        plt.tight_layout()
        plt.savefig("%s%snm/%snm_fitted-peak-positions.png" % (Fig_dir, laser, laser), dpi=300)
        plt.show()
        
        # report average positions for material non-specific peaks
        print()
        peaks = [450, 678, 745, 1530]
        vals = np.zeros((len(peaks)+1, len(raman['fitted_peaks'])))
        colors = [Color_list[Materials.index(raman['material'][i2]) % len(Color_list)] for i2 in range(len(raman['material']))]
        for i, peak in enumerate(peaks):
            vals[i] = np.asarray([find_max(raman['raman_shift'][i2], raman['y_av_sub_norm'][i2], peak-10, peak+10)[1] for i2 in range(len(raman['material']))])
            result = [i2 for i2 in range(len(raman['fitted_peaks'])) if len(raman['fitted_peaks'][i2]['center']) > 0]
            print(f"{laser}nm {peak}: {len(result)} spectra")
            print("    ", np.shape(vals[i]))
            if len(result) > 0:
                result = [i2 for i2 in result if np.amin(np.abs(raman['fitted_peaks'][i2]['center'] - peak)) < 10]
                if len(result) > 2:
                    temp = []
                    for i2 in result:
                        ### print("        ", raman['fitted_peaks'][i]['center'])
                        ### print("            ", raman['fitted_peaks'][i]['center'][np.argmin(np.abs(raman['fitted_peaks'][i]['center'] - peak))])
                        temp.append(raman['fitted_peaks'][i2]['center'][np.argmin(np.abs(raman['fitted_peaks'][i2]['center'] - peak))])
                    print(f"    mean position {np.mean(temp):0.1f} +/- {np.std(temp):0.1f} cm-1")
                else:
                    print(f"    no spectra with peaks in range")
                plt.title(f"{laser}nm unknown peak at {peak} cm-1")
                count = 0
                offset = 0.4
                for i2 in result:
                    material = raman['material'][i2]
                    colour = Material_colors[Materials.index(material) % len(Material_colors)]
                    plt.plot(raman['raman_shift'][i2], raman['y_av_sub_norm'][i2]+offset*count, colour)
                    count += 1
                plt.xlim(400, 1800)
                plt.xlabel("Raman Shift (cm${-1}$)")
                plt.ylabel("Normalised Intensity (offset)")
                plt.yticks([])
                plt.savefig("%s%snm/%snm_unknown-peak_%0.fcm.png" % (Fig_dir, laser, laser, peak), dpi=300)
                plt.show()
                
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.title("%s vs %s cm-1" % (peaks[0], peaks[1]))
        plt.scatter(vals[0], vals[1], c=colors)
        plt.subplot(132)
        plt.title("%s vs %s cm-1" % (peaks[0], peaks[2]))
        plt.scatter(vals[0], vals[2], c=colors)
        plt.subplot(133)
        plt.title("%s vs %s cm-1" % (peaks[1], peaks[2]))
        plt.scatter(vals[1], vals[2], c=colors)
        plt.show()
            
# plot fitted peak positions by material and laser
if Plot_fit_summary == True and Plot_material_summary == True:
    for material in materials:
        print(material)
        if material in Materials:
            colour = Material_colors[Materials.index(material) % len(Material_colors)]
        else:
            colour = Material_colors[count % len(Material_colors)]
        if material in Peak_positions.keys():
            ref_peaks = np.unique(Peak_positions[material])
        else:
            ref_peaks = np.unique(np.concatenate([Peak_positions[key] for key in Peak_positions.keys()]))
        mfc = colour
        samples = [raman['sample'][i] for i in range(len(raman['sample'])) if raman['material'][i] == material]
        if len(samples) == 0:
            print("    no matching %s nm data for %s" % (laser, material))
        else:
            plt.figure(figsize=(10,8))
            ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
            ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
            ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
            ax1.set_title(f"{material} Fitted Peak Positions")
            ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
            ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
            ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
            if plot_fwhm == True:
                ax1.set_ylabel("FWHM (cm$^{-1}$)")
                ax2.set_ylabel("FWHM (cm$^{-1}$)")
                ax3.set_ylabel("FWHM (cm$^{-1}$)")
            else:
                ax1.set_yticks([])
                ax2.set_yticks([])
                ax3.set_yticks([])
            ax1.set_xlim(400, 4000)
            ax2.set_xlim(400, 1800)
            ax3.set_xlim(2800, 3100)
            if material == 'unassigned':
                for key in ['PE', 'PP', 'PPE', 'PS']:
                    for peak in Peak_positions[key]:
                        ax1.axvline(peak, color=Material_colors[Materials.index(key) % len(Material_colors)], linestyle=':', zorder=4)
                        ax2.axvline(peak, color=Material_colors[Materials.index(key) % len(Material_colors)], linestyle=':', zorder=4)
                        ax3.axvline(peak, color=Material_colors[Materials.index(key) % len(Material_colors)], linestyle=':', zorder=4)
            elif material in Peak_positions.keys():
                for peak in Peak_positions[material]:
                    ax1.axvline(peak, color=colour, linestyle=':', zorder=4)
                    ax2.axvline(peak, color=colour, linestyle=':', zorder=4)
                    ax3.axvline(peak, color=colour, linestyle=':', zorder=4)
            labels = []
            for laser in Lasers:
                count = 0
                marker = Marker_list[Lasers.index(laser)]
                for sample in samples:
                    imax = get_best_sample_spectrum(raman, sample, laser, goal='peaks', x_start=500, x_end=3300)
                    if imax != None:
                        peaks = raman['fitted_peaks'][imax]
                        if len(peaks['center']) > 0:
                            alphas = peaks['amplitude'] / np.amax(peaks['amplitude'])
                            if plot_fwhm == True:
                                y = peaks['fwhm']
                                y_err = peaks['fwhm standard error']
                            else:
                                y = np.full_like(peaks['center'], -count)
                                y_err = np.zeros_like(peaks['center'])
                            if plot_errorbars == True:
                                for i2 in range(len(peaks['center'])):
                                    ax1.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                                    ax2.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                                    ax3.errorbar(peaks['center'][i2], y[i2], xerr=peaks['center standard error'], yerr=y_err[i2], ecolor=colour, mec=colour, mfc=mfc, elinewidth=1., linewidth=0., alpha=alphas[i2])
                            label = "_%snm" % str(laser)
                            if label not in labels:
                                labels.append(label)
                                label = "%snm" % str(laser)
                            for i2 in range(len(peaks['center'])):
                                if np.amin(np.abs(peaks['center'][i2] - ref_peaks)) <= 10:
                                    mfc = colour
                                else:
                                    mfc = 'w'
                                ax1.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                                ax2.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                                ax3.scatter(peaks['center'][i2], y[i2], c=mfc, edgecolors=colour, alpha=alphas[i2])
                            count += 1
            plt.figtext(0.15, 0.5, "%s/%s samples successfully fitted" % (count, len(np.unique(raman['sample'][np.where(raman['material'] == material)]))))
            ### ax1.legend()
            plt.tight_layout()
            plt.savefig("%sby material/%s/%s_fitted-peak-positions.png" % (Fig_dir, material, material), dpi=300)
            plt.show()

"""
# ==================================================
# run PCA and K-means clustering on spectra
# ==================================================
"""

SNV = True
normalise = True
first_deriv = True

debug = False

if Do_PCA == True:
    print()
    print("running PCA...")
    
    if normalise == True:
        text = '1st-deriv-of-norm'
    else:
        text = '1st-deriv'
        
    # create array containing all relevant parameters, for ease of filtering
    spec_temp = np.vstack((raman['laser_wavelength'], raman['laser_power'], raman['accumulations'], raman['exposure_time'], raman['x_start'], raman['x_end'])).transpose()
        
    for laser in Lasers:
        # get parameters for spectrum filtering
        power = Target_params[str(laser)]['laser_power']
        accum = Target_params[str(laser)]['accumulations']
        for region in Target_params[str(laser)]['regions']:
            x_start, x_end = region
            if x_end >= 3000 and laser == '785':
                exposure = 1.
            else:
                exposure = Target_params[str(laser)]['exposure_time']
            print()
            print("PCA spectrum parameters: %s nm, %3.1f%% %2.fx%2.1fs, %4.f-%4.f cm-1" % (laser, power, accum, exposure, x_start, x_end))
            params = np.array([laser, power, accum, exposure])
            spec_count = 0
            samples = []
            for i, spec in enumerate(spec_temp):
                if np.all(spec[0:4] == params) and spec[4] <= x_start and spec[5] >= x_end:
                    spec_count += 1
                    if raman['sample'][i] not in samples:
                        samples.append(raman['sample'][i])
            print("    matching spectra:", spec_count)
            print("    matching samples:", len(samples))
            print()
            print("Running PCA on %s nm, region %0.f to %0.f cm-1" % (laser, x_start, x_end))
            x_temp = np.linspace(x_start, x_end, 2*int(x_end-x_start)+1)
            y_temp = []
            mat_temp = []
            sample_temp = []
            id_temp = []
            
            # check how many samples have matching spectra
            samples = []
            spectra = []
            spec_indices = []
            for sample in Sample_IDs:
                # find only those spectra for this sample that match the defined parameters
                specsort = np.ravel(np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser) & (raman['x_start'] <= x_start) & (raman['x_end'] >= x_end) & (raman['laser_power'] == power) & (raman['accumulations'] == accum) & (raman['exposure_time'] == exposure)))
                if debug == True:
                    print("    %s:" % sample, len(specsort), [int(item.split("_")[-1]) for item in raman['ID'][specsort]])
                if len(specsort) > 0:
                    specsort = specsort[np.argsort(raman['points'][specsort])]
                    samples.append(raman['sample'][specsort[-1]])
                    spectra.append(raman['ID'][specsort[-1]])
                    spec_indices.append(specsort[-1])
                    if debug == True:
                        print("        %16s: %2.f points, %3.1f%%, %2.fx%2.1fs, %4.f-%4.f cm-1" % (raman['ID'][specsort[-1]], raman['points'][specsort[-1]], raman['laser_power'][specsort[-1]], raman['accumulations'][specsort[-1]], raman['exposure_time'][specsort[-1]], raman['x_start'][specsort[-1]], raman['x_end'][specsort[-1]]))
            spec_indices = np.asarray(spec_indices)
            print()
            print("    matching samples found:", len(samples))
            if debug == True:
                print("        ", samples)
            print("    matching spectra found:", len(spectra))
            if debug == True:
                print("        ", spectra)

            if len(spec_indices) > 0:
                # select standard X values for interpolation
                print()
                # build Y value array
                for i in spec_indices:
                    # for each matching spectrum
                    sample = raman['sample'][i]
                    assignment = get_chemicalID(sample_database, sample)
                    # interpolate y data based on x_temp
                    y = np.interp(x_temp, raman['raman_shift'][i], raman['y_av_sub'][i])
                    if debug == True:
                        print("    %s Y arrays:" % raman['ID'][i], np.shape(y))
                    # do normalisation if necessary
                    if SNV == True:
                        y = (y - np.mean(y)) / np.std(y)
                    elif normalise == True:
                        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
                    if first_deriv == True:
                        # Calculate first derivative using a Savitzky-Golay filter
                        y_deriv = savgol_filter(y, 25, polyorder = 5, deriv=1)
                        y_temp.append(y_deriv)
                    else:
                        y_temp.append(y)
                    sample_temp.append(sample)
                    if assignment[0] in ['HDPE', 'MDPE', 'LDPE']:
                        mat_temp.append("PE")
                    else:
                        mat_temp.append(assignment[0])
                    id_temp.append(raman['ID'][i])
                        
            sample_temp = np.asarray(sample_temp)
            mat_temp = np.asarray(mat_temp)
            x_temp = np.asarray(x_temp)
            y_temp = np.asarray(y_temp)
            print()
            print("    PCA x,y array shapes:", np.shape(x_temp), np.shape(y_temp))
            print("        materials:", np.unique(mat_temp))

            if len(sample_temp) > 6:
                # proceed with PCA
                temp = pd.DataFrame(y_temp, columns=x_temp, index=sample_temp)
                ### print(temp.info)

                pca = PCA(n_components=int(np.amin([6,len(sample_temp)])))
                principalComponents = pca.fit_transform(temp)
                principal_frame = pd.DataFrame(data=principalComponents, columns=['principal component '+str(i+1) for i in range(0, pca.n_components_)])

                print("    features:", pca.n_features_)
                print("    components:", pca.n_components_)
                print("    samples:", pca.n_samples_)
                print("    Explained variation per principal component:")
                for i in range(0, pca.n_components_):
                    print("        component %d: %0.3f" % (i+1, pca.explained_variance_ratio_[i]))

                final_frame = pd.concat([principal_frame, pd.DataFrame(mat_temp, columns=['material']), pd.DataFrame(sample_temp, columns=['sample']), pd.DataFrame(id_temp, columns=['ID'])], axis=1)

                # set up figure
                plt.figure(figsize=(12, 6))
                # ax1: PCA coordinates plot, by material
                ax1 = plt.subplot(121)
                ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
                ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
                ax1.set_title('%s nm, %s-%s cm$^{-1}$' % (laser, x_start, x_end))
                labels = []
                for material, colour in zip(Materials, Material_colors):
                    indices = final_frame['material'] == material
                    if np.any(indices) == True:
                        labels.append(material+" (%s)" % (np.count_nonzero(mat_temp == material)))
                        ax1.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], edgecolors=colour, c=colour)
                ax1.grid()

                # ax3: loading spectra for each component
                ax3 = plt.subplot(122)
                ax3.set_xlabel("Frequency (cm$^{-1}$)")
                ax3.set_ylabel("Variance")
                ax3.set_xlim(x_start, x_end)
                for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
                    comp = comp * var  # scale component by its variance explanation power
                    ax3.plot(x_temp, np.cumsum(comp), label="component %s" % (i+1))
                        
                # finish figure
                ax1.legend(labels)
                ax2.legend(labels)
                ax3.legend()
                plt.tight_layout()
                plt.savefig("%s%snm/PCA_%s_%0.f-%0.fcm.png" % (Fig_dir, laser, text, x_start, x_end))
                plt.show()
                    
                if Do_clustering == True:
                    # run K-means
                    print()
                    print("running K-means clustering on %s nm, region %0.f-%0.f cm-1" % (laser, x_start, x_end))

                    n_clusters = 4
                    print("    number of clusters:", n_clusters)

                    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
                    kmeans.fit(principalComponents)

                    for cluster in range(0, n_clusters):
                        print("        spectra in cluster %2d: %0.f, centered at: (%0.3f, %0.3f)" % (cluster, np.count_nonzero(kmeans.labels_ == cluster), kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1]))
                        for material in np.unique(mat_temp):
                            sort = np.logical_and(final_frame['material'] == material, kmeans.labels_ == cluster)
                            if np.count_nonzero(sort) > 0:
                                print("            %3.f%% %s (%0.f/%0.f %s spectra)" % (100*np.count_nonzero(sort)/np.count_nonzero(kmeans.labels_ == cluster), material, np.count_nonzero(sort), np.count_nonzero(final_frame['material'] == material), material))
                                print(", ".join(final_frame['sample'][sort]))

                    plt.figure(figsize=(18,6))
                    ax1 = plt.subplot(131)
                    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
                    ax3 = plt.subplot(133)
                    ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
                    ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
                    ax1.set_title('PCA of %s nm, %s-%s cm$^{-1}$' % (laser, x_start, x_end))
                    ax2.set_title('K-Means Clustering')
                    ax3.set_title("Cluster Mean Spectra")
                    ax3.set_xlim(x_start, x_end)
                    
                    # plot data points by material type
                    labels = []
                    for material, colour in zip(Materials, Material_colors):
                        indices = final_frame['material'] == material
                        if np.any(indices) == True:
                            labels.append(material+" (%s)" % (np.count_nonzero(mat_temp == material)))
                            mfc = colour
                            ax1.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], edgecolors=colour, c=mfc)
                    ax1.legend(labels)
                    ax1.grid(zorder=3)

                    # print clustering breakdown of each material group
                    print()
                    for material in np.unique(mat_temp):
                        indices = final_frame['material'] == material
                        bar_offset = 0
                        if np.any(indices) == True:
                            print()
                            print("%s: %0.f spectra in %0.f clusters:" % (material, np.count_nonzero(indices), len(np.unique(kmeans.labels_[indices]))))
                            for i in np.unique(kmeans.labels_[indices]):
                                sort = np.logical_and(final_frame['material'] == material, kmeans.labels_ == i)
                                if np.count_nonzero(sort) > 0:
                                    print("    cluster %2d: %4d spectra (%3.f%%)" % (i, np.count_nonzero(sort), 100.*np.count_nonzero(sort)/np.count_nonzero(indices)))
                    print()

                    # plot clusters as scatter
                    labels = []
                    for cluster in range(0, n_clusters):
                        # iterate over clusters
                        colour = Color_list[cluster]
                        indices = kmeans.labels_ == cluster
                        if np.any(indices) == True:
                            ax2.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], c=colour, label="cluster %0.f (%0.f)" % (cluster, np.count_nonzero(indices)), zorder=1)
                            labels.append("cluster" + str(cluster))
                            ### ax2.scatter(kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1], edgecolors=colour, c='w', marker='s', label="cluster %0.f (%0.f)" % (cluster, np.count_nonzero(indices)), zorder=2)
                            ax3.plot(x_temp, np.cumsum(np.mean(y_temp[indices], axis=0)), colour, label="cluster %0.f" % (cluster))
                            
                    # finish figure
                    ax2.legend()
                    ax2.grid(zorder=3)
                    ax3.legend()
                    plt.tight_layout()
                    plt.savefig("%s%snm/PCA_%s_%0.f-%0.fcm_clusters.png" % (Fig_dir, laser, text, x_start, x_end))
                    plt.show()
                    
                # plot PCA of each material coloured by max intensity
                if Plot_material_summary == True:
                    print()
                    print("plotting PCA with points coloured by intensity")
                    plt.figure(figsize=(12,6))
                    ax1 = plt.subplot(121)
                    ax2 = plt.subplot(122)
                    ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
                    ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
                    ax1.set_title('PCA of %s nm, %s-%s cm$^{-1}$' % (laser, x_start, x_end))
                    ax2.set_xlim(x_start, x_end)
                    # plot data points by material type
                    labels = []
                    for material, color in zip(Materials, Material_colors):
                        indices = final_frame['material'] == material
                        if np.any(indices) == True:
                            mean_y = np.cumsum(np.mean(y_temp[indices], axis=0))
                            max_i = np.argmax(mean_y)
                            max_intensities = np.cumsum(y_temp[indices], axis=1)[:,max_i]
                            max_intensities /= np.amax(max_intensities)
                            for i, factor in zip(final_frame.index[indices], max_intensities):
                                if factor > 1:
                                    factor = 1.
                                elif factor < 0:
                                    factor = 0.
                                colour = mpl.colors.to_rgba(color)
                                colour = (colour[0]*factor, colour[1]*factor, colour[2]*factor)
                                ### print(colour)
                                ax1.scatter(final_frame.loc[i, 'principal component 1'], final_frame.loc[i, 'principal component 2'], c=colour)
                            ax2.plot(x_temp, mean_y, color, label=material)
                            ax2.axvline(x_temp[max_i], c=color, linestyle=':')
                    ax1.legend(labels)
                    ax2.legend()
                    ax1.grid(zorder=3)
                    plt.show()
                    
print()
for laser in Lasers:
    print("samples measured using %s nm:" % laser, len(np.unique(raman['sample'][np.where(raman['laser_wavelength'] == laser)])))
    for material in Materials:
        print("    %s: %s" % (material, len(np.unique(raman['sample'][np.where((raman['laser_wavelength'] == laser) & (raman['material'] == material))]))))
                    
print()
print()
print("total spectra processed:", len(raman['ID']))
for laser in Lasers:
    print("    %s nm:" % laser, len(raman['ID'][np.where(raman['laser_wavelength'] == laser)]))
print("total samples processed:", len(np.unique(raman['sample'])))
for laser in Lasers:
    print("    %s nm:" % laser, len(np.unique(raman['sample'][np.where(raman['laser_wavelength'] == laser)])))
print()
print("DONE")
