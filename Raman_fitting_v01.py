"""
# ==================================================
This script takes processed Raman spectra of plastic samples and does peak fitting to determine peak positions, widths, and intensities.

Dependencies: Raman_processing script (to process raw spectra and output baselined spectra as .csv files)

Data import relies on processed spectra files using the following naming convention:
    MeasurementDate_SampleID_LaserWavelength_LaserPower_Accumulations-x-ExposureTime_SpectralRange_PointsAveraged_spectra.csv
This can be amended by changing lines 483-502

Fitted peak parameters are saved to files in the Output folder for further analysis/interpretation.

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
Plot_sample_summary = True          # produce a figure for each unique sample

# specify which processes to run
Find_peaks = True                   # find peaks by maxima
Fit_peaks = True                    # fit peaks using mathematical functions
if Fit_peaks == True:
    Fit_function = 'PV'             # choose from 'G' (Gaussian), 'L' (Lorentzian), or 'PV' (pseudo-Voigt)
    Starting_peaks = 'auto'         # 'manual' to use user-defined positions or 'auto' to use results from Find Peaks process
    Peak_positions = []             # list of peak positions for fitting if Starting_peaks == 'manual'
    
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
# functions for fitting a baseline

def f_linbase(x, *params):
    # function for generating a linear baseline
    a, b = params
    y = a*x + b
    return y

def f_polybase(x, *params):
    # function for generating an exponential baseline
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y

def smooth_f(y, window_length, polyorder):
    # function for smoothing data based on Savitsky-Golay filtering
    if window_length % 2 != 1:
        window_length += 1
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth

def average_list(x, y, point_list, window, debug=False):
    # function for taking a set of user-defined points and creating arrays of their average x and y values
    if debug == True:
        print("        ", point_list)
    x_averages = np.zeros_like(point_list, dtype=float)
    y_averages = np.zeros_like(point_list, dtype=float)
    point_num = 0
    for i in range(np.size(point_list)):
        point_num += 1
        x_averages[i], y_averages[i] = local_average(x, y, point_list[i], window)
        if debug == True:
            print("        point", str(point_num), ": ", x_averages[i], y_averages[i])
    return x_averages, y_averages

def local_average(x, y, x_0, w):
    # function for finding the average position from a set of points, centered on 'x_0' with 'w' points either side
    center_ind = np.argmin(np.absolute(x - x_0))
    start_ind = center_ind - w
    end_ind = center_ind + w
    if start_ind < 0:
        start_ind = 0
    if end_ind > len(y)-1:
        end_ind = -1
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average

def linbase_fit(x_averages, y_averages, debug=False):
    # function for fitting selected average data-points using a linear function
    guess = [1., 0.]
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_linbase, x_averages, y_averages, p0=guess)
    if debug == True:
        print("        fitted parameters: ", fit_coeffs)
    return fit_coeffs, fit_covar

def polybase_fit(x_averages, y_averages, max_order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(max_order):
        guess = np.zeros((int(max_order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polybase, x_averages, y_averages, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

def baseline(x, y, x_list, base='poly', max_order=15, window=25, find_minima=True, debug=False, plot=False, name=None):
    # calculate baseline and subtract it
    if plot == True:
        # create figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        if name != None:
            ax1.set_title("%s:\nBaseline-Corrected Raman Spectrum" % (str(name)))
        else:
            ax1.set_title("Baseline-Corrected Raman Spectrum")
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
    # smooth data for fitting
    y_s = smooth_f(y, 5, 3)
    # find point positions and fit with curve
    if find_minima == True:
        points = []
        for point in x_list:
            points.append(find_min(x, y_s, point-window, point+window)[0])
    else:
        points = x_list
    # create arrays of average values for each point +/-5 pixels
    x_averages, y_averages = average_list(x, y_s, points, 5, debug=debug)
    if base in ['lin', 'line', 'linear']:
        # fit the points with a linear function
        fit_coeffs, fit_covar = linbase_fit(x_averages, y_averages, debug=debug)
        y_basefit = f_linbase(x, *fit_coeffs)
    elif base in ['exp', 'exponential']:
        # fit the points with a exponential function
        fit_coeffs, fit_covar = expbase_fit(x_averages, y_averages, debug=debug)
        y_basefit = f_expbase(x, *fit_coeffs)
    elif base in ['sin', 'sine', 'sinewave']:
        # fit the points with a sinewave function
        fit_coeffs, fit_covar = sinbase_fit(x_averages, y_averages, debug=debug)
        y_basefit = f_sinbase(x, *fit_coeffs)
    else:
        # fit the points with a polynomial function
        fit_coeffs, fit_covar = polybase_fit(x_averages, y_averages, max_order=max_order, debug=debug)
        y_basefit = f_polybase(x, *fit_coeffs)
    y_B = y - y_basefit
    if plot == True:
        # plot points and fit in ax1
        ax1.plot(x, y, 'k')
        ax1.plot(x_averages, y_averages, 'or', label='points')
        ax1.plot(x, y_basefit, 'r', label='baseline')
        # plot before and after in ax2
        ax2.plot(x, y, 'k', label='before')
        ax2.plot(x, y_B, 'b', label='after')
        ax2.legend(loc=1)
        y_max = find_max(x_averages, y_averages, np.amin(x_list), np.amax(x_list))[1]
        y_min = find_min(x_averages, y_averages, np.amin(x_list), np.amax(x_list))[1]
        ax1.set_ylim(y_min-0.5*(y_max-y_min), y_min+1.5*(y_max-y_min))
        ax1.set_xlim(np.amin(x_list)-100, np.amax(x_list)+100)
        y_max = find_max(x, y, np.amin(x_list), np.amax(x_list))[1]
        y_min = find_min(x, y, np.amin(x_list), np.amax(x_list))[1]
        ax2.set_ylim(-0.5*(y_max-y_min), y_min+1.5*(y_max-y_min))
        ax2.set_xlim(np.amin(x_list)-100, np.amax(x_list)+100)
        plt.tight_layout()
        ### plt.savefig("%s/%s_av_base.png" % (sample_figdirs[i], samples[i]), dpi=300)
        ### plt.savefig("%s/%s_av_base.svg" % (sample_figdirs[i], samples[i]), dpi=300)
        plt.show()
    return y_B

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

def G_curve(x, params):
    model = np.zeros_like(x)
    gradient = params['gradient']
    intercept = params['intercept']
    A = params['amplitude']
    mu = params['center']
    sigma = params['sigma']
    model += A * np.exp(-0.5*(x - mu)**2/(sigma**2)) + gradient*x + intercept
    return model

def multiPV_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        eta = params['eta_%s' % i]
        model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiPV_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        eta = params['eta_%s' % i]
        model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiL_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * (sigma**2)/((x - mu)**2 + sigma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiL_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * (sigma**2)/((x - mu)**2 + sigma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiG_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiG_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def peak_fit_script(x, y, maxima, window=10., max_sigma=30., function='gaussian', vary_baseline=True, debug=False):
    # script for fitting a set of maxima with a pre-defined function (defaults to gaussian)
    params = lmfit.Parameters()
    params.add('gradient', value=0., vary=vary_baseline)
    params.add('intercept', value=np.amin(y), vary=vary_baseline)
    for i in range(0, len(maxima)):
        y_max = x[np.argmin(np.absolute(y - maxima[i]))]
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-window, max=maxima[i]+window)
        params.add('amplitude_%s' % i, value=y_max, min=0.)
        if function.lower() in ['pv', 'pseudo-voigt', 'psuedo-voigt']:
            params.add('sigma_%s' % i, value=10., min=2., max=2.*max_sigma)
            params.add('eta_%s' % i, value=0.5, min=0., max=1.)
        elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
            params.add('width_%s' % i, value=10., min=2., max=2*max_sigma)
            params.add('rounding_%s' % i, value=5., min=0.)
        elif function.lower() in ['l', 'lorentz', 'lorentzian']:
            params.add('sigma_%s' % i, value=10., min=2., max=2*max_sigma)
        else:
            params.add('sigma_%s' % i, value=10., min=2., max=2.*max_sigma)
    if debug == True:
        print("        initial parameters:")
        print(params.pretty_print())
    if function.lower() in ['pv', 'pseudo-voigt', 'psuedo-voigt']:
        fit_output = lmfit.minimize(multiPV_fit, params, args=(x, y, maxima))
        fit_curve = multiPV_curve(x, fit_output.params, maxima)
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
        fit_output = lmfit.minimize(multiFD_fit, params, args=(x, y, maxima))
        fit_curve = multiFD_curve(x, fit_output.params, maxima)
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        fit_output = lmfit.minimize(multiL_fit, params, args=(x, y, maxima))
        fit_curve = multiL_curve(x, fit_output.params, maxima)
    else:
        fit_output = lmfit.minimize(multiG_fit, params, args=(x, y, maxima))
        fit_curve = multiG_curve(x, fit_output.params, maxima)
    if debug == True:
        print("        fit status: ", fit_output.message)
        print("        fitted parameters:")
        print(fit_output.params.pretty_print())
    return fit_output, fit_curve

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
print("total spectra with material assignments:", spec_count)

# generate colourmap for total energy used to acquire spectrum
cmin = 0.75*np.amin(raman['total_energy'])
cmax = 1.33*np.amax(raman['total_energy'])
norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
cmap.set_array([])

"""
# ==================================================
# Find Peaks by searching for maxima
# ==================================================
"""

x_start, x_end = (400, 3500)                # region to assess for peak detection
window = 10                                 # minimum acceptable separation between peaks in pixels
rel_intensity_threshold = 0.3               # minimum intensity for peak detection vs spectrum max
SNR_threshold = 8.                          # minimum signal:noise ratio for peak detection

show_plots = False                          # show plots in viewer
plot_peak_summary = True                    # produce figure summarising peak results for each spectrum

if Find_peaks == True:
    raman['detected_peaks'] = []
    
    print()
    print("detecting peaks by maxima...")
    
    for i in range(0, len(raman['sample'])):
        # find peaks in each sample average spectrum
        sample = raman['sample'][i]
        laser = raman['laser_wavelength'][i]
        x = raman['raman_shift'][i]
        y = raman['y_av_sub'][i]
        raman["detected_peaks"].append([])
        
        print()
        print(sample)
        print("    ", np.shape(x), np.shape(y))
        
        # get min/max of spectrum
        y_min = np.amin(raman['y_av_sub'][i][np.where((x_start <= raman['raman_shift'][i])& (raman['raman_shift'][i] <= x_end))])
        y_max = np.amax(raman['y_av_sub'][i][np.where((x_start <= raman['raman_shift'][i])& (raman['raman_shift'][i] <= x_end))])
        
        # smooth data before searching for maxima
        y_temp = savgol_filter(y, 11, 3)
        
        # find maxima that are above relative intensity threshold
        maxima = find_maxima(x, y_temp, window, rel_intensity_threshold)
        print("    %s maxima detected:" % sample, len(maxima[0]), maxima[0])
        
        # determine background noise level using 3 regions (500-600, 1800-1900, 3200-3300)
        noise = [np.std(y[np.where((500 <= x) & (x <= 600))]), np.std(y[np.where((1800 <= x) & (x <= 1900))])]
        if raman['x_end'][i] >= 3300:
            noise.append(np.std(y[np.where((3200 <= x) & (x <= 3300))]))
        noise = np.mean(noise)
        print("    mean noise level: %0.1f counts" % noise)
        
        # only pass maxima that are above SNR threshold
        maxima_pass = [[],[]]
        for i2 in range(0, len(maxima[0])):
            if maxima[1,i2] > float(SNR_threshold) * noise:
                maxima_pass[0].append(maxima[0,i2])
                maxima_pass[1].append(maxima[1,i2])
                ax1.text(maxima[0,i2], maxima[1,i2]+0.05*(y_max-y_min), "%0.f" % maxima[0,i2], rotation=90, va='bottom', ha='left')
        maxima_pass = np.asarray(maxima_pass)
        
        # create plot for peak detection
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nPeak Detection" % raman['ID'][i])
        ax1.axhline(rel_intensity_threshold*y_max, color='b', linestyle=':', label='minimum intensity')
        plt.plot(x, y, 'k', label='data')
        ax1.axhline(float(SNR_threshold)*noise, color='r', linestyle=':', label='minimum SNR')
        ax1.plot(maxima[0], maxima[1], 'ro', label='fail  (%0.0f)' % (len(maxima[0])-len(maxima_pass[0])))
        # plot detected maxima
        ax1.plot(maxima_pass[0], maxima_pass[1], 'bo', label='pass (%0.0f)' % len(maxima_pass[0]))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Average Intensity (counts)")
        # create second y axis for SNR values
        ax2 = ax1.twinx()
        ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
        ax2.set_ylabel("SNR")
        ax1.legend()
        plt.tight_layout()
        plt.savefig("%s%s_detected-peaks.png" % (raman['fig_dir'][i], raman['ID'][i]), dpi=300)
        plt.show()
        
        # rearrange into ascending order 
        maxima_pass = maxima_pass[:,np.argsort(maxima_pass[0])]
            
        # add maxima positions, intensities to array
        raman["detected_peaks"][-1] = maxima_pass
        
"""
# ==================================================
# attempt peak fitting
# ==================================================
"""

if Fit_peaks == True:
    raman['fitted_peaks'] = []
    
    print()
    print("fitting peaks...")
    
    for i in range(0, len(raman['sample'])):
        # fit peaks for each sample average spectrum
        sample = raman['sample'][i]
        laser = raman['laser_wavelength'][i]
        x = raman['raman_shift'][i]
        y = raman['y_av_sub'][i]
        raman['fitted_peaks'].append({})
        
        print()
        print(sample)
        print("    ", np.shape(x), np.shape(y))
        
        # select starting peak positions
        if Starting_peaks == 'manual':
            # use user-defined list of starting peak positions
            maxima_pass = np.zeros((2, len(Peak_positions)))
            for i2 in range(len(Peak_positions)):
                maxima_pass[:,i2] = [x[np.argmin(np.abs(x - Peak_positions[i2]))], y[[np.argmin(np.abs(x - Peak_positions[i2]))]]]
        else:
            # assume automatic, use results from peak detection
            maxima_pass = raman['detected_peaks'][i]
            
        print()
        print("    %s peaks to fit:" % np.size(maxima_pass, axis=1), list(maxima_pass[0]))
            
        # group peaks that are close together into regions for fitting
        regions = []
        window = 200.           # minimum distance between 2 peaks to count as being in the same region
        fitted_regions_x = []
        fitted_regions_y = []
        
        if len(maxima_pass[0]) > 1:
            print()
            print("    generating fit regions for %s..." % raman['ID'][i])
            # start with first peak (in ascending order)
            temp = [maxima_pass[0][0]]
            
            for i2 in range(1, len(maxima_pass[0])):
                local_max = maxima_pass[0][i2]
                if local_max - temp[-1] < window:
                    # if gap between this peak and the last is <100% of the window size, add to current group
                    temp.append(local_max)
                else:
                    # otherwise create region for previous group and start new temp group
                    x_start = np.amin(temp) - window
                    x_end = np.amax(temp) + window
                    # round up/down to nearest 10
                    x_start = np.floor(x_start/10.)*10.
                    x_end = np.ceil(x_end/10.)*10.
                    # check region doesn't fall outside the data range
                    if np.amin(raman['raman_shift'][i]) > x_start:
                        x_start = np.amin(raman['raman_shift'][i])
                    if np.amax(raman['raman_shift'][i]) < x_end:
                        x_end = np.amax(raman['raman_shift'][i])
                    regions.append([x_start, x_end])
                    temp = [local_max]
                    
            # then resolve final region
            x_start = np.amin(temp) - window
            x_end = np.amax(temp) + window
            
            # round x_start, x_end to nearest 10. for reporting purposes
            x_start = np.floor(x_start/10.)*10.
            x_end = np.ceil(x_end/10.)*10.
            
            # check region doesn't fall outside the data range
            if np.amin(raman['raman_shift'][i]) > x_start:
                x_start = np.amin(raman['raman_shift'][i])
            if np.amax(raman['raman_shift'][i]) < x_end:
                x_end = np.amax(raman['raman_shift'][i])
            regions.append([x_start, x_end])
            print("        region %s: %0.f - %0.f cm-1, %s peaks" % (len(regions), x_start, x_end, len(temp)))
        
        elif len(maxima_pass[0]) == 1:
            print()
            print("    generating single fit region for %s..." % raman['ID'][i])
            # create single region around only peak
            local_max = maxima_pass[0][0]
            x_start = local_max - window
            x_end = local_max + window
            # round up/down to nearest 10
            x_start = np.floor(x_start/10.)*10.
            x_end = np.ceil(x_end/10.)*10.
            # check region doesn't fall outside the data range
            if np.amin(raman['raman_shift'][i]) > x_start:
                x_start = np.amin(raman['raman_shift'][i])
            if np.amax(raman['raman_shift'][i]) < x_end:
                x_end = np.amax(raman['raman_shift'][i])
            regions.append([x_start, x_end])
            ### print("            ", [local_max])
            print("        region %s: %0.f - %0.f cm-1, 1 peak" % (len(regions), x_start, x_end))
        
        else:
            print("    cannot continue with fit, no peaks found!")
        
        
        # create temporary data array for fitting results
        fitted_peaks = {'function': [], 'centers': [], 'amplitudes': [], 'fwhm': [], 'centers_err': [], 'amplitudes_err': [], 'fwhm_err': []}
        
        if len(regions) > 0:
            # proceed with fitting each region separately using positions from maxima_pass
            print()
            print("    at least one region found, proceeding with peak fit...")
            
            # set up arrays depending on specified function
            if Fit_function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
                function = 'pv'
                fitted_peaks['sigmas'] = []
                fitted_peaks['sigmas_err'] = []
                fitted_peaks['etas'] = []
                fitted_peaks['etas_err'] = []
            elif Fit_function.lower() in ['l', 'lorentz', 'lorentzian']:
                function = 'l'
                fitted_peaks['gammas'] = []
                fitted_peaks['gammas_err'] = []
            else:
                function = 'g'
                fitted_peaks['sigmas'] = []
                fitted_peaks['sigmas_err'] = []
            print("        fitting function:", function.upper())
                
            for i2 in range(0, len(regions)):
                # slice data to region
                reg_start = regions[i2][0]
                reg_end = regions[i2][1]
                print()
                print("        region %s: %0.f - %0.f cm-1" % (i2+1, reg_start, reg_end))
                input_peaks = maxima_pass[0][np.where((reg_start < maxima_pass[0]) & (maxima_pass[0] < reg_end))]
                print("            %s peaks found:" % len(input_peaks), input_peaks)
                x_slice = x[np.where((reg_start <= x) & (x <= reg_end))]
                y_slice = y[np.where((reg_start <= x) & (x <= reg_end))]
                
                # proceed with fit
                fit_output, fit_curve = peak_fit_script(x_slice, y_slice, input_peaks, function=function, window=10., max_sigma=30.)
                
                # plot results
                plt.figure(figsize=(8,6))
                # ax1: results of fit
                ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
                ax1.set_title("%s\n%0.f-%0.f cm$^{-1}$ %s Peak Fitting" % (raman['ID'][i], reg_start, reg_end, Fit_function))
                ax1.set_ylabel("Average Intensity")
                # ax2: residuals
                ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
                ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax2.set_ylabel("Residual")
                # histogram of residuals
                ax3 = plt.subplot2grid((4,5), (3,4))
                ax3.set_yticks([])
                # determine y limits for residual, hist plots
                y_min = np.amin(y_slice-fit_curve)
                y_max = np.amax(y_slice-fit_curve)
                res_min = y_min - 0.1*(y_max-y_min)
                res_max = y_max + 0.1*(y_max-y_min)
                ax2.set_ylim(res_min, res_max)
                # plot input data and residuals
                ax1.plot(x_slice, y_slice, 'k')
                ax2.plot(x_slice, y_slice-fit_curve, 'k')
                ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
                # plot individual peak fits
                x_temp = np.linspace(reg_start, reg_end, 10*len(x_slice))
                for i2 in range(0, len(input_peaks)):
                    # add parameters to storage array
                    fitted_peaks['function'].append(function)
                    for prop in ['center', 'amplitude', 'sigma', 'gamma', 'eta', 'width', 'round']:
                        key = prop+"_%s" % i2
                        if key in fit_output.params.keys():
                            fitted_peaks["%ss" % prop].append(fit_output.params[key].value)
                            if fit_output.params[key].stderr != None:
                                fitted_peaks[prop+"s_err"].append(fit_output.params[key].stderr)
                            else:
                                fitted_peaks[prop+"s_err"].append(0.)
                    
                    if function == 'l':
                        # for Fermi-Dirac functions, FWHM is defined as 2*gamma
                        fitted_peaks['fwhm'].append(2. * fit_output.params['gamma_%s' % i2].value)
                        if fit_output.params['gamma_%s' % i2].stderr != None:
                            fitted_peaks['fwhm_err'].append(2. * fit_output.params['gamma_%s' % i2].stderr)
                        else:
                            fitted_peaks['fwhm_err'].append(0.)
                    else:
                        # for pseudo-voigt and gaussian functions, FWHM is defined as 2*sqrt(2)*sigma
                        fitted_peaks['fwhm'].append(2.355 * fit_output.params['sigma_%s' % i2].value)
                        if fit_output.params['sigma_%s' % i2].stderr != None:
                            fitted_peaks['fwhm_err'].append(2.355 * fit_output.params['sigma_%s' % i2].stderr)
                        else:
                            fitted_peaks['fwhm_err'].append(0.)
                        
                    # plot and report peak positions
                    plt.figtext(0.78, 0.93-0.08*i2, "Center %s: %.1f" % (i2+1, fitted_peaks['centers'][-1]))
                    plt.figtext(0.78, 0.9-0.08*i2, " FWHM %s: %.1f" % (i2+1, fitted_peaks['fwhm'][-1]))
                    ax1.axvline(fit_output.params["center_%s" % i2], color='k', linestyle=':')
                    # create function curve for plotting
                    params = lmfit.Parameters()
                    params.add('gradient', value=fit_output.params["gradient"])
                    params.add('intercept', value=fit_output.params["intercept"])
                    params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i2])
                    params.add('center_0', value=fit_output.params["center_%s" % i2])
                    if function == 'pv':
                        params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                        params.add('eta_0', value=fit_output.params["eta_%s" % i2])
                        peak_curve = multiPV_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    elif function == 'fd':
                        params.add('width_0', value=fit_output.params["width_%s" % i2])
                        params.add('round_0', value=fit_output.params["round_%s" % i2])
                        peak_curve = multiFD_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    elif function == 'l':
                        params.add('gamma_0', value=fit_output.params["gamma_%s" % i2])
                        peak_curve = multiL_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    else:
                        params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                        peak_curve = multiG_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    ax1.plot(x_temp, peak_curve, 'b:')
                # plot total peak fit
                if function == 'pv':
                    total_curve = multiPV_curve(x_temp, fit_output.params, input_peaks)
                elif function == 'fd':
                    total_curve = multiFD_curve(x_temp, fit_output.params, input_peaks)
                elif function == 'l':
                    total_curve = multiL_curve(x_temp, fit_output.params, input_peaks)
                else:
                    total_curve = multiG_curve(x_temp, fit_output.params, input_peaks)
                ax1.plot(x_temp, total_curve, 'b--')
                # finish fitting figure
                y_max = np.amax(y_slice)
                ax1.set_xlim(reg_start, reg_end)
                ax1.set_ylim(np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)]), 1.2*y_max)
                plt.savefig("%s/%s_%0.f-%0.fcm_fit.png" % (raman['fig_dir'][i], raman['ID'][i], reg_start, reg_end), dpi=300)
                plt.savefig("%s/%s_%0.f-%0.fcm_fit.svg" % (raman['fig_dir'][i], raman['ID'][i], reg_start, reg_end), dpi=300)
                plt.show()
                
        # add results (as datadict) to data storage array
        for key in fitted_peaks.keys():
            raman['fitted_peaks'][i][key] = np.asarray(fitted_peaks[key])
            print(key, np.shape(raman['fitted_peaks'][i][key]))
        
        # save fit parameters to file
        if len(fitted_peaks['centers']) > 0:
            print()
            print("saving peak fit data to file")
            save_data = []
            header = []
            for prop in ['centers', 'amplitudes', 'fwhm', 'sigmas', 'gammas', 'etas', 'widths', 'rounds']:
                name = prop
                if prop[-1] == 's':
                    name = prop[:-1]
                if prop in fitted_peaks.keys():
                    save_data.append(fitted_peaks[prop])
                    save_data.append(fitted_peaks[prop+"_err"])
                    header.append(name)
                    header.append(name+" standard error")
            save_data = np.vstack(save_data)
            save_name = "%s_%s_%0.f-%0.fcm_%s-point" % (raman['measurement_date'][i].strftime("%Y-%m-%d"), raman['ID'][i], raman['x_start'][i], raman['x_end'][i], raman['points'][i])
            # save peak data to output folder
            np.savetxt("%s%snm/by sample/%s/%s_%s-fit-parameters.csv" % (Out_dir, raman['laser_wavelength'][i], raman['sample'][i], save_name, function.upper()), save_data.transpose(), header=", ".join(header), delimiter=', ')

    print()
    print("sample array:", len(raman['sample']))
    print("fitted peak array:" , len(raman['fitted_peaks']))
    
    if Plot_sample_summary == True:
        for sample in np.unique(raman['sample']):
            result = np.ravel(np.where(raman['sample'] == sample))
            print()
            print("plotting summary of fitted peak positions for %s" % sample)
            if len(result) > 1:
                plt.figure(figsize=(8,2+len(result)/2))
                ax1 = plt.subplot2grid((1,3), (0,0), colspan=2)
                ax2 = plt.subplot2grid((1,3), (0,2))
                plt.suptitle("%s: Fitted Peak Positions vs Laser Energy" % sample)
                ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
                ax1.set_xlim(400, 2000)
                ax2.set_xlim(2600, 3000)
                ax1.set_yticks([])
                ax2.set_yticks([])
                labels = []
                count = 0
                for laser in Lasers:
                    result = np.ravel(np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser)))
                    for i in result:
                        text = "%snm %3.1f%% %0.fx%0.1fs" % (raman['laser_wavelength'][i], raman['laser_power'][i], raman['accumulations'][i], raman['exposure_time'][i])
                        print("%s: %s %s, %0.1f mJ" % (i, sample, text, raman['total_energy'][i]))
                        print("    peaks fitted:", len(raman['fitted_peaks'][i]['centers']))
                        if len(raman['fitted_peaks'][i]['centers']) > 0:
                            dose = raman['total_energy'][i]
                            color = Laser_colors[All_Lasers.index(laser)]
                            for i2 in range(0, len(raman['fitted_peaks'][i]['centers'])):
                                alpha = raman['fitted_peaks'][i]['amplitudes'][i2]/np.amax(raman['fitted_peaks'][i]['amplitudes'])
                                label = "_%snm" % raman['laser_wavelength'][i]
                                if label not in labels and alpha == 1.:
                                    labels.append(label)
                                    label = "%snm" % raman['laser_wavelength'][i]
                                ax1.scatter(raman['fitted_peaks'][i]['centers'][i2], count, c=color, alpha=alpha)
                                ax2.scatter(raman['fitted_peaks'][i]['centers'][i2], count, c=color, alpha=alpha, label=label)
                        count += 1
                    count += 5
                plt.tight_layout(rect=(0, 0, 1, 0.95))
                ax2.legend()
                plt.savefig("%sby sample/%s/%s_fitted_positions.png" % (Fig_dir, sample, sample))
                plt.show()
            
            

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
