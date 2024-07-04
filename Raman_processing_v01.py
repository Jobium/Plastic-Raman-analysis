"""
# ==================================================
This script processes Raman spectra of plastic fragments measured at any given wavelength
    1) averaging together of spectra for the same sample taken using the same settings
    2) baseline subtraction

This script is designed to accept files with the following naming convention:
    MeasurementDate(YYYY-MM-DD)_SampleID_LaserWavelength_LaserPower_Accumulations-x-ExposureTime_SpectralRange_SpectrumNumber_OptionalNotes.txt
This can be amended by changing lines 483-502


All processed spectra are saved to files in the Output folder for further analysis/interpretation.

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
Data_dir = './data/'

# directory of sample info database containing list of sample names and their material assignments
Database_dir = './data/Database.csv'

# directory of folder where figures will be saved
Fig_dir = './figures/'

# directory of folder where processed data will be saved
Out_dir = './output/'

# specify which figures to generate
Plot_sample_summary = True          # produce a figure for each unique sample
Plot_material_summary = True        # produce a figure for each unique material

# specify which processes to run
Baseline_by_section = True          # baseline each section of spectrum separately (necessary for spectra with discontinuities)
Baseline_individual_spectra = True  # baseline indidivual spectra then get the average

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
        end_ind = len(y)-1
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average

def expbase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using an exponential function
    guess = [1., 0.05]
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_expbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters: ", fit_coeffs)
    return fit_coeffs, fit_covar

def sinebase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using a sine function
    guess = [(np.amax(y_averages)-np.amin(y_averages))/2., (np.amax(x_averages)-np.amin(x_averages))/4., 0., np.mean(y_averages)]
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_sinbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters: ", fit_coeffs)
    return fit_coeffs, fit_covar

def linbase_fit(x_averages, y_averages, sigma, debug=False):
    # function for fitting selected average data-points using a linear function
    guess = [1., 0.]
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_linbase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters: ", fit_coeffs)
    return fit_coeffs, fit_covar

def polybase_fit(x_averages, y_averages, sigma, max_order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(max_order):
        guess = np.zeros((int(max_order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polybase, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

def baseline(x, y, x_list, raman_i, base='poly', max_order=15, window=25, find_minima=True, fixed_ends=True, debug=False, plot=False, saveplot=True, name=None, splits=[], baseline_by_section=False):
    global raman, Color_list
    # calculate baseline and subtract it
    if plot == True or saveplot == True:
        # create figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)
        ax1.plot(x, y, 'k')
        ax2 = plt.subplot(212)
        if name == None:
            name = raman['ID'][raman_i]
        ax1.set_title("%s:\nBaseline-Corrected Raman Spectrum" % str(name))
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
    if debug == True:
        print(raman_i, raman['ID'][raman_i], name, raman['fig_dir'][raman_i])
    # smooth data for fitting
    y_s = smooth_f(y, 5, 3)
    # divide up data into sections (default: 1)
    if baseline_by_section == True and len(splits) > 0:
        fixed_ends = True
        section_splits = [np.argmin(np.abs(x - float(point))) for point in splits if point > np.amin(x) and point < np.amax(x)]
        if debug == True:
            print("    spectrum split at points", ", ".join([str(split) for split in splits]))
    else:
        section_splits = 1
        if debug == True:
            print("    no spectrum splitting")
    x_splits = np.split(x, section_splits)
    y_splits = np.split(y_s, section_splits)
    if debug == True:
        print("    x,y split arrays:", np.shape(x_splits), np.shape(y_splits))
    fitted_baseline = []
    for i, x_slice, y_slice in zip(range(0, len(x_splits)), x_splits, y_splits):
        if debug == True:
            print("        section %s x, y arrays:" % i, np.shape(x_slice), np.shape(y_slice))
            print("        section %s x range: %0.1f - %0.1f cm-1" % (i, np.amin(x_slice), np.amax(x_slice)))
        # trim x_list to section
        x_list_trim = np.asarray(x_list)[np.where((np.amin(x_slice) <= np.asarray(x_list)) & (np.asarray(x_list) < np.amax(x_slice)))]
        if find_minima == True:
            # find minimum value within window around each point in x_list_trim 
            points = []
            for point in x_list_trim:
                points.append(find_min(x_slice, y_slice, point-window, point+window)[0])
        else:
            points = x_list_trim
        # create arrays of average values for each point +/-5 pixels
        x_averages, y_averages = average_list(x_slice, y_slice, points, 5, debug=debug)
        # add fixed first and last points if applicable, and create sigma array for point weighting
        sigma = np.ones_like(y_averages)
        if fixed_ends == True:
            for index in [5, -6]:
                x_0, y_0 = local_average(x_slice, y_slice, x_slice[index], 5)
                x_averages = np.append(x_averages, x_0)
                y_averages = np.append(y_averages, y_0)
                sigma = np.append(sigma, 0.1)
        # sort x_list into ascending order
        sort = np.argsort(x_averages)
        x_averages = x_averages[sort]
        y_averages = y_averages[sort]
        sigma = sigma[sort]
        # attempt to fit section data using specified base function
        while True:
            try:
                if base in ['lin', 'linear']:
                    fit_coeffs, fit_covar = linbase_fit(x_averages, y_averages, sigma, debug=debug)
                    basefit = f_linbase(x_slice, *fit_coeffs)
                elif base in ['exp', 'exponential']:
                    fit_coeffs, fit_covar = expbase_fit(x_averages, y_averages, sigma, debug=debug)
                    basefit = f_expbase(x_slice, *fit_coeffs)
                elif base in ['sin', 'sine', 'sinewave']:
                    fit_coeffs, fit_covar = sinebase_fit(x_averages, y_averages, sigma, debug=debug)
                    basefit = f_sinebase(x_slice, *fit_coeffs)
                else:
                    if max_order > len(y_averages)-1:
                        order = len(y_averages)-1
                    else:
                        order = max_order
                    fit_coeffs, fit_covar = polybase_fit(x_averages, y_averages, sigma, max_order=order, debug=debug)
                    basefit = f_polybase(x_slice, *fit_coeffs)
                fitted_baseline.append(basefit)
                if plot == True or saveplot == True:
                    ax1.scatter(x_averages, y_averages, c=Color_list[i % len(Color_list)])
                    ax1.plot(x_slice, basefit, Color_list[i % len(Color_list)])
                break
            except Exception as e:
                if debug == True:
                    print("    something went wrong! Exception:", e)
                    print("        attempting another fit with reduced polynomial order...")
                try:
                    if order - 1 > 1:
                        order -= 1
                    fit_coeffs, fit_covar = polybase_fit(x_averages, y_averages, sigma, max_order=order, debug=debug)
                    basefit = f_polybase(x_slice, *fit_coeffs)
                    fitted_baseline.append(basefit)
                    if plot == True or saveplot == True:
                        ax1.scatter(x_averages, y_averages, c=Color_list[i % len(Color_list)])
                        ax1.plot(x_slice, basefit, Color_list[i % len(Color_list)])
                    break
                except Exception as e:
                    if debug == True:
                        print("        something went wrong again! Exception:", e)
                        print("            taking minimum for flat baseline")
                    fitted_baseline.append(np.full_like(y_slice, np.amin(y_slice)))
                    if plot == True or saveplot == True:
                        ax1.scatter(x_averages, y_averages, c=Color_list[i % len(Color_list)])
                        ax1.plot(x_slice, np.full_like(y_slice, np.amin(y_slice)), Color_list[i % len(Color_list)])
                    break
    # concatenate sections into a single baseline
    fitted_baseline = np.concatenate(fitted_baseline)
    # subtract baseline from data
    y_sub = y - fitted_baseline
    if plot == True or saveplot == True:
        # plot baseline-subtracted spectrum
        ax2.axhline(0., c='k', linestyle=':')
        ax2.plot(x, y_sub, 'k', label='baselined')
        ax2.legend(loc=1)
        x_list_trim = np.asarray(x_list)[np.where((np.amin(x) <= np.asarray(x_list)) & (np.asarray(x_list) <= np.amax(x)))]
        y_min = find_min(x, fitted_baseline, np.amin(x_list_trim), np.amax(x_list_trim))[1]
        y_max = find_max(x, fitted_baseline, np.amin(x_list_trim), np.amax(x_list_trim))[1]
        ax1.set_ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
        ax1.set_xlim(np.amin(x_list_trim)-100, np.amax(x_list_trim)+100)
        y_min = find_min(x, y_sub, np.amin(x_list_trim), np.amax(x_list_trim))[1]
        y_max = find_max(x, y_sub, np.amin(x_list_trim), np.amax(x_list_trim))[1]
        ax2.set_ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
        ax2.set_xlim(np.amin(x_list_trim)-100, np.amax(x_list_trim)+100)
        plt.minorticks_on()
        plt.tight_layout()
        if saveplot == True:
            if debug == True:
                print("    saving baseline figure to %s%s_baseline.png" % (raman['fig_dir'][raman_i], name))
            plt.savefig("%s%s_baseline.png" % (raman['fig_dir'][raman_i], name), dpi=300)
            plt.savefig("%s%s_baseline.svg" % (raman['fig_dir'][raman_i], name), dpi=300)
        if plot == True:
            plt.show()
        else:
            plt.close()
    return y_sub

# ==================================================
# functions for finding peaks

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

files = sorted(glob.glob("%s%snm/*.txt" % (Data_dir, str(Laser_Wavelength))))

print()
print("    %s files found" % len(files))

# ==================================================
# import files into empty datadict

raman = {
    'ID': [],
    'measurement_date': [],
    'sample': [],
    'wavelength': [],
    'spec_num': [],
    'spec_type': [],
    'points': [],
    'raman_shift': [],
    'y': [],
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
    'fig_dir': [],
    'out_dir': [],
}

count = 0
for file in files:
    if 'background' not in file:
        count += 1
        filename = file.split("/")[-1][:-4]
        print()
        print("    %i:" % count, filename)
        while True:
            try:
                print("        importing %s..." % file)
                # extract info from filename
                sample = filename.split("_")[1]
                print("        sample:", sample)
                date = datetime.datetime.strptime(filename.split("_")[0], "%Y-%m-%d")
                print("        measured on:", date.strftime("%YYYY-%m-%d"))
                
                # extract measurement info
                print("        spectrum properties:")
                power, accum, exposure, grating = (1,1,1,0)
                for prop in filename.split("_"):
                    if prop[-2:] == 'nm':
                        laser = int(prop[:-2])
                        print("            laser: %0.f nm" % laser)
                    if prop[-1] == '%':
                        power = float(prop[:-1].replace(",", "."))
                        print("            laser power: %0.1f %%" % power)
                    if prop[-1] == 's':
                        if len(prop.split("x")) == 2:
                            accum = int(prop.split("x")[0])
                            exposure = float(prop.split("x")[1][:-1].replace(",", "."))
                            print("            %s accumulations of %s seconds" % (accum, exposure))
                            
                # calculate total input laser energy
                total_energy = float(exposure)*(float(power)/100.)*Laser_power[str(laser)]
                print("        total energy used: %0.1f mJ" % total_energy)
                print("            in photons:", photon_count(float(laser), float(total_energy)))
                # import data array
                spec = np.genfromtxt(file, delimiter='\t')
                print("            spec array:", np.shape(spec))
                print("                nan check:", np.any(np.isnan(spec)))
                print("                inf check:", np.any(np.isinf(spec)))
                if np.size(spec, axis=1) < np.size(spec, axis=0):
                    # transpose array if necessary
                    spec = spec.transpose()
                    print("            transposed spec array:", np.shape(spec))
                # determine if file contains single or multiple spectra
                if np.size(spec, axis=0) > 2:
                    # file contains more than 2 columns, assume all other columns are additional spectra
                    if len(prop.split("-")[0].split('x')) > 1:
                        points = prop.split("-")[0].split('x')[0]*prop.split("-")[0].split('x')[1]
                    else:
                        points = int(prop.split("-")[0])
                    spec_type = 'map'
                    print("            spec type: %s-point map" % points)
                    print("                expected spec length: %0.1f pixels" % (np.size(spec, axis=1)/points))
                    if np.size(spec, axis=1) % points == 0:
                        # split array according to number of points
                        x_temp = np.stack((np.split(spec[2], points)), axis=0)[0]
                        y_temp = np.stack((np.split(spec[3], points)), axis=0)
                    else:
                        print("            array length cannot be divided by recorded number of points!")
                        print("        aborting import...")
                        break
                else:
                    points = 1
                    spec_type = 'point'
                    print("        spec type: single point")
                    x_temp = spec[0]
                    y_temp = spec[1][np.newaxis,:]
                # sort array by raman shift in ascending order (fixes arrays that are in descending order)
                sort = np.argsort(x_temp)
                x_temp = x_temp[sort]
                y_temp = y_temp[:,sort]
                print("        temp x array:", np.shape(x_temp))
                print("        temp y array:", np.shape(y_temp))
                # extract raman shift range from values
                x_start = np.rint(np.amin(x_temp))
                x_end = np.rint(np.amax(x_temp))
                print("        shift range: %0.2f - %0.2f cm-1" % (x_start, x_end))
                wavelengths = shift2wavelength(x_temp, excitation=laser)
                print("            in nm: %0.1f - %0.1f nm" % (np.amin(wavelengths), np.amax(wavelengths)))
                # remove data below edge filter
                edge = 0.
                if np.any(x_temp <= edge):
                    x_temp = x_temp[x_temp >= edge]
                    y_temp = y_temp[:, x_temp >= edge]
                    print("            trimming points below edge filter")
                    x_start = np.rint(np.amin(x_temp))
                    x_end = np.rin(np.amax(x_temp))
                    print("            new shift range: %0.f - %0.f cm-1" % (x_start, x_end))
                    wavelengths = shift2wavelength(x_temp, excitation=laser)
                    print("                in nm: %0.f - %0.f nm" % (np.amin(wavelengths), np.amax(wavelengths)))
                
                # check if spectrum metadata matches any other spectra
                id_temp = "%s_%snm_%.2f%%_%0.fx%.2fs" % (sample, str(laser), power, accum, exposure)
                print("        assigned ID: %s" % id_temp)
                result = np.ravel(np.where(np.asarray(raman['ID']) == id_temp))
                print("        %0.f imported spectra with matching parameters:" % len(result), np.asarray(raman['points'])[result])
                
                if len(result) > 0:
                    success = False
                    for i in result:
                        if success == False:
                            # parameters are identical to already imported spectrum, combine together
                            print("        comparing to spec numbers", raman['spec_num'][i])
                            print("            x array sizes: %s vs %s" % (np.size(x_temp),  np.size(raman['raman_shift'][i])))
                            print("             start values: %s vs %s" %  (x_temp[0], raman['raman_shift'][i][0]))
                            print("               end values: %s vs %s" %  (x_temp[-1], raman['raman_shift'][i][-1]))
                            print("              value check:", np.count_nonzero(x_temp == raman['raman_shift'][i]))
                            if np.size(x_temp) != np.size(raman['raman_shift'][i]):
                                print("            size of raman shift arrays do not match! %s vs %s" % (np.size(x_temp), np.size(raman['raman_shift'][i])))
                                if np.absolute(np.size(x_temp) - np.size(raman['raman_shift'][i])) < 5:
                                    print("                arrays differ in size by %s pixels, fixing by interpolating data" % (np.size(x_temp)-np.size(raman['raman_shift'][i])))
                                    y_temp = np.interp(raman['raman_shift'][i], x_temp, np.ravel(y_temp))[np.newaxis,:]
                                    x_temp = np.copy(raman['raman_shift'][i])
                                else:
                                    print("                arrays differ in size by %s pixels, cannot import!" % (np.size(x_temp)-np.size(raman['raman_shift'][i])))
                                    break
                            elif np.any(x_temp != raman['raman_shift'][i]):
                                print("            values in raman shift arrays do not match exactly!")
                                if np.count_nonzero(x_temp != raman['raman_shift'][i]) < 5:
                                    print("                %%0.f pixels are erroneous, fixing by interpolating data" % (np.count_nonzero(x_temp != raman['raman_shift'][i]), i))
                                    y_temp = np.interp(raman['raman_shift'][i], x_temp, np.ravel(y_temp))[np.newaxis,:]
                                    x_temp = np.copy(raman['raman_shift'][i])
                                else:
                                    print("                %0.f pixels are erroneous, cannot safely import!" % (np.count_nonzero(x_temp != raman['raman_shift'][i])))
                                    break
                            print("        good match, combining with spec numbers", raman['spec_num'][i])
                            raman['spec_num'][i].append(count)
                            raman['y'][i] = np.vstack((raman['y'][i], y_temp))
                            raman['points'][i] += 1
                            success = True
                
                else:
                    # parameters are unique, add to array as new object
                    print("        adding data to array")
                    raman['ID'].append(id_temp)
                    raman['measurement_date'].append(date)
                    raman['spec_num'].append([count])
                    raman['spec_type'].append(spec_type)
                    raman['points'].append(points)
                    raman['sample'].append(sample)
                    raman['wavelength'].append(wavelengths)
                    raman['raman_shift'].append(x_temp)
                    raman['y'].append(y_temp)
                    raman['x_start'].append(np.round(x_start, -1))
                    raman['x_end'].append(np.round(x_end, -1))
                    raman['x_range'].append(np.round(x_end, -1)-np.round(x_start, -1))
                    raman['laser_wavelength'].append(str(laser))
                    raman['laser_power'].append(power)
                    raman['accumulations'].append(accum)
                    raman['exposure_time'].append(exposure)
                    raman['total_energy'].append(total_energy)
                    raman['photon_count'].append(photon_count(laser, total_energy))
                    
                    # check if fragment has material assignment in fragment database:
                    assignment = get_chemicalID(sample_database, sample)
                    if assignment != '':
                        raman['material'].append(assignment)
                    else:
                        raman['material'].append("unassigned")
                    
                    # create folders for figures, data output
                    fig_dir = "%sby sample/%s/" % (Fig_dir, sample)
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)
                    fig_dir = "%s%snm/by sample/%s/" % (Fig_dir, str(laser), sample)
                    if not os.path.exists(fig_dir):
                        os.makedirs(fig_dir)
                    raman['fig_dir'].append(fig_dir)
                    out_dir = "%s%snm/by sample/%s/" % (Out_dir, str(laser), sample)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    raman['out_dir'].append(out_dir)
                print("        import finished!")
                break
            except Exception as e:
                print("        something went wrong!", e)
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
print("combined spectra found in Raman data:  ", len(raman['ID']))
print("individual spectra found in Raman data:", np.sum(raman['points']))

# report how many spectra found for each fragment at each wavelength, by date
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
# average raw spectra and generate raw figures
# ==================================================
"""

debug = True

print()
print("processing data...")

raman['y_av'] = []
raman['y_std'] = []
raman['y_av_norm'] = []
raman['y_std_norm'] = []

for i in range(0, len(raman['sample'])):
    raman['y_av'].append(np.mean(raman['y'][i], axis=0))
    raman['y_std'].append(np.std(raman['y'][i], axis=0))
    y_max = find_max(raman['raman_shift'][i], raman['y_av'][i], 400, 4000)[1]
    raman['y_av_norm'].append(raman['y_av'][i]/y_max)
    raman['y_std_norm'].append(raman['y_std'][i]/y_max)
    
# sort available spectra by energy and spectral range
indices = np.lexsort((raman['total_energy'], raman['x_range']))
    
offset = 0.5
for sample in Sample_IDs:
    for laser in Lasers:
        # sort available spectra for this sample and wavelength by energy and spectral range
        result = indices[np.where((raman['sample'][indices] == sample) & (raman['laser_wavelength'][indices] == laser))]
        count = 0
        if len(result) > 0:
            # plot figure of raw spectra for each sample
            plt.figure(figsize=(8,2+0.25*len(result)))
            plt.title("%s: %s nm" % (sample, laser))
            for i in result:
                plt.plot(raman['raman_shift'][i], raman['y_av_norm'][i]+count*offset, c=cmap.to_rgba(raman['total_energy'][i]), label=raman['ID'][i].split("_")[-1])
                count += 1
            plt.xlabel("Raman Shift (cm$^{-1}$)")
            plt.ylabel("Normalised Intensity (offset)")
            plt.yticks([])
            plt.legend()
            plt.savefig("%s%s_%snm_raw-spectra.png" % (raman['fig_dir'][i], sample, laser), dpi=300)
            plt.show()

# plot all raw spectra for each sample that cover x_start - x_end range
x_start, x_end = (None, None)   # set to (None, None) to show all possible ranges
offset = 0.5                    # offset in y axis between spectra
for sample in Sample_IDs:
    # count how many spectra match this sample and range (for determining plot size)
    if x_start != None and x_end != None:
        result = indices[np.where((raman['sample'] == sample) & (x_start >= raman['x_start']) & (raman['x_end'] >= x_end))]
    else:
        result = indices[np.where(raman['sample'] == sample)]
    plt.figure(figsize=(8,3+0.5*len(result)))
    plt.title("%s\nRaw Spectra by Wavelength" % (sample))
    labels = []
    count = 0
    for laser in Lasers:
        # count how many spectra match this sample and wavelength and range
        if x_start != None and x_end != None:
            result = indices[np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser) & (x_start >= raman['x_start']) & (raman['x_end'] >= x_end))]
        else:
            result = indices[np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser))]
        if len(result) > 0:
            for i in result:
                # plot spectrum
                if laser not in labels:
                    plt.plot(raman['raman_shift'][i], raman['y_av_norm'][i]+count*offset, c=Laser_colors[All_Lasers.index(laser)], label="%s nm" % laser)
                    labels.append(laser)
                else:
                    plt.plot(raman['raman_shift'][i], raman['y_av_norm'][i]+count*offset, c=Laser_colors[All_Lasers.index(laser)])
                count += 1
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Normalised Intensity (offset)")
    plt.yticks([])
    plt.ylim(-0.5*offset, 0.5+count*offset)
    if x_start != None and x_end != None:
        plt.xlim(x_start, x_end)
    plt.legend()
    plt.savefig("%sby sample/%s/%s_raw-spectra.png" % (Fig_dir, sample, sample), dpi=300)
    plt.show()

"""
# ==================================================
# do polynomial baseline subtraction
# ==================================================
"""

print()
print("baselining spectra...")

debug = True

# manually define Raman shifts for splitting spectra
baseline_splits = {'532': [2700, 3000], '638': [1900, 2100], '785': [1350, 1410, 1850, 2200, 2600]}

raman['y_sub'] = []
raman['y_sub_norm'] = []
raman['y_sub_av'] = []
raman['y_sub_std'] = []
raman['y_sub_av_norm'] = []
raman['y_sub_std_norm'] = []
raman['y_av_sub'] = []
raman['y_std_sub'] = []
raman['y_av_sub_norm'] = []
raman['y_std_sub_norm'] = []
for i in range(0, len(raman['sample'])):
    print()
    print("%s, %s nm" % (raman['ID'][i], raman['laser_wavelength'][i]))
    # decide which set of baseline points to use
    if raman['laser_wavelength'][i] == '532':
        x_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500, 1550, 1610, 1800, 1900, 2000, 2050, 2100, 2200, 2300, 2400, 2500, 2600, 2650, 2700, 2780, 2800, 2850, 2900, 3000, 3100, 3150, 3250, 3300, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
        order = 11
    elif raman['laser_wavelength'][i] in ['633', '638']:
        x_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500, 1550, 1610, 1700, 1750, 1800, 1850, 1880, 1900, 1920, 1940, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2200, 2300, 2400, 2600, 2800, 3000, 3150, 3300, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
        order = 21
        if raman['sample'][i] in ['BW-2021-110-P020']:
            x_list = [100, 200, 300, 400, 415, 450, 500, 550, 600, 650, 675, 700, 725, 750, 775, 800, 825, 850, 900, 950, 1000, 1050, 1150, 1250, 1400, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2200, 2300, 2400, 2600, 2800, 3000, 3150, 3300, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
            order = 51
    elif raman['laser_wavelength'][i] == '785':
        x_list = [100, 230, 300, 400, 415, 450, 500, 550, 650, 700, 800, 900, 1050, 1200, 1250, 1300, 1330, 1360, 1375, 1390, 1405, 1410, 1480, 1500, 1550, 1610, 1700, 1800, 1900, 1950, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2800, 3000, 3150, 3300, 3350, 3400, 3450, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
        order = 11
    else:
        x_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500, 1550, 1610, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2600, 2800, 3150, 3300, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
        order = 11
        
    x_list = np.asarray(x_list)
    # proceed with baseline subtraction of average spectrum
    if Baseline_by_section == True and raman['laser_wavelength'][i] in baseline_splits.keys():
        splits = baseline_splits[raman['laser_wavelength'][i]]
        # stepped spectrum, baseline each step separately then recombine
        print("    baselining spectrum in sections")
        name = raman['ID'][i]
        y_sub = baseline(raman['raman_shift'][i], raman['y_av'][i], x_list, i, base='poly', max_order=order, fixed_ends=True, name=raman['ID'][i], plot=True, saveplot=True, debug=True, splits=splits, baseline_by_section=True)
        # add sub data to arrays
        raman['y_av_sub'].append(np.copy(y_sub))
        raman['y_std_sub'].append(raman['y_std'][i])
        y_max = find_max(raman['raman_shift'][i], raman['y_av_sub'][i], raman['x_start'][i], raman['x_end'][i])[1]
        raman['y_av_sub_norm'].append(raman['y_av_sub'][i]/y_max)
        raman['y_std_sub_norm'].append(raman['y_std'][i]/y_max)
            
        # baseline individual spectra if required
        if Baseline_individual_spectra == True and raman['points'][i] > 1:
            raman['y_sub'].append(np.empty_like(raman['y'][i]))
            raman['y_sub_norm'].append(np.empty_like(raman['y'][i]))
            for i2 in range(raman['points'][i]):
                y_sub = baseline(raman['raman_shift'][i], raman['y'][i][i2], x_list, i, base='poly', max_order=order, fixed_ends=True, name=raman['ID'][i] + "_spec%s" % str(raman['spec_num'][i][i2]).zfill(4), plot=False, saveplot=True, debug=False, splits=splits, baseline_by_section=True)
                #  add sub data to arrays
                raman['y_sub'][i][i2] = np.copy(y_sub)
                y_max = find_max(raman['raman_shift'][i], raman['y_sub'][i][i2], raman['x_start'][i], raman['x_end'][i])[1]
                raman['y_sub_norm'][i][i2] = raman['y_sub'][i][i2]/y_max
            
            plt.title("%s:\nAverage of %s baselined spectra" % (raman['sample'][i], raman['points'][i]))
            plt.fill_between(raman['raman_shift'][i], np.mean(raman['y_sub'][i], axis=0)-np.std(raman['y_sub'][i], axis=0), np.mean(raman['y_sub'][i], axis=0)+np.std(raman['y_sub'][i], axis=0), color='k', alpha=0.1, linewidth=0.)
            plt.plot(raman['raman_shift'][i], np.mean(raman['y_sub'][i], axis=0), 'k')
            plt.xlim(raman['x_start'][i], raman['x_end'][i])
            plt.show()
            # add averaged sub data to arrays
            raman['y_sub_av'].append(np.mean(raman['y_sub'][i], axis=0))
            raman['y_sub_std'].append(np.std(raman['y_sub'][i], axis=0))
            y_max = find_max(raman['raman_shift'][i], raman['y_sub_av'][i], raman['x_start'][i], raman['x_end'][i])[1]
            raman['y_sub_av_norm'].append(raman['y_sub_av'][i]/y_max)
            raman['y_sub_std_norm'].append(raman['y_sub_std'][i]/y_max)
        else:
            # only one spectrum, pass duplicate data instead
            raman['y_sub'].append(raman['y_av_sub'][i])
            raman['y_sub_norm'].append(raman['y_av_sub_norm'][i])
            raman['y_sub_av'].append(raman['y_av_sub'][i])
            raman['y_sub_std'].append(raman['y_std_sub'][i])
            raman['y_sub_av_norm'].append(raman['y_av_sub_norm'][i])
            raman['y_sub_std_norm'].append(raman['y_std_sub_norm'][i])
            
    else:
        # no sections, baseline entire spectrum at once
        x_list_trim = np.sort(x_list[np.where((raman['x_start'][i]+10 <= x_list) & (x_list <= raman['x_end'][i]-10))])
        # reduce polynomial order if it exceeds number of points-1
        if len(x_list_trim)-1 < order:
            order = len(x_list_trim)-1
        print("    %0.f-%0.f cm-1, x_list:" % (raman['x_start'][i], raman['x_end'][i]), len(x_list_trim), x_list_trim)
        print("    nan check:", np.any(np.isnan(raman['raman_shift'][i])), np.any(np.isnan(raman['y_av'][i])), np.any(np.isnan(x_list_trim)))
        print("    inf check:", np.any(np.isinf(raman['raman_shift'][i])), np.any(np.isinf(raman['y_av'][i])), np.any(np.isinf(x_list_trim)))
        # subtract baseline and add to array
        raman['y_av_sub'].append(baseline(raman['raman_shift'][i], raman['y_av'][i], x_list_trim, i, base='poly', max_order=order, fixed_ends=True, name=raman['ID'][i], plot=True, saveplot=True, debug=debug))
        y_max = find_max(raman['raman_shift'][i], raman['y_av_sub'][i], raman['x_start'][i], raman['x_end'][i])[1]
        raman['y_av_sub_norm'].append(raman['y_av_sub'][i]/y_max)
        raman['y_std_sub'].append(raman['y_std'][i])
        raman['y_std_sub_norm'].append(raman['y_std'][i]/y_max)

        # baseline individual spectra if required
        if Baseline_individual_spectra == True and raman['points'][i] > 1:
            raman['y_sub'].append(np.empty_like(raman['y'][i]))
            raman['y_sub_norm'].append(np.empty_like(raman['y'][i]))
            x_list_trim = np.sort(np.asarray(x_list)[np.where((raman['x_start'][i]+10 <= x_list) & (x_list <= raman['x_end'][i]-10))])
            # reduce polynomial order if it exceeds number of points-1
            if len(x_list_trim)-1 < order:
                order = len(x_list_trim)-1
            print("%s points:" % raman['ID'][i], raman['points'][i], np.shape(raman['raman_shift'][i]), np.shape(raman['y'][i]))
            for i2 in range(raman['points'][i]):
                # subtract baseline and add to array
                raman['y_sub'][i][i2] = baseline(raman['raman_shift'][i], raman['y'][i][i2], x_list_trim, i, base='poly', max_order=order, fixed_ends=True, name=raman['ID'][i]+"_spec%s" % raman['spec_num'][i][i2], plot=False, saveplot=True, debug=False)
                y_max = find_max(raman['raman_shift'][i], raman['y_sub'][i][i2], raman['x_start'][i], raman['x_end'][i])[1]
                raman['y_sub_norm'][i][i2] = raman['y_sub'][i][i2]/y_max
            
            # plot average of baselines vs baseline of average
            plt.title("%s:\nAverage of %s baselined spectra" % (raman['sample'][i], raman['points'][i]))
            plt.fill_between(raman['raman_shift'][i], np.mean(raman['y_sub'][i], axis=0)-np.std(raman['y_sub'][i], axis=0), np.mean(raman['y_sub'][i], axis=0)+np.std(raman['y_sub'][i], axis=0), color='k', alpha=0.1, linewidth=0.)
            plt.plot(raman['raman_shift'][i], np.mean(raman['y_sub'][i], axis=0), 'k')
            plt.xlim(raman['x_start'][i], raman['x_end'][i])
            plt.show()
            
            # get average of baselined spectra
            raman['y_sub_av'].append(np.mean(raman['y_sub'][i], axis=0))
            raman['y_sub_std'].append(np.std(raman['y_sub'][i], axis=0))
            raman['y_sub_av_norm'].append(raman['y_av_sub'][i]/np.amax(raman['y_av_sub'][i]))
            raman['y_sub_std_norm'].append(raman['y_std_sub'][i]/np.amax(raman['y_av_sub'][i]))
        else:
            # only one spectrum
            raman['y_sub'].append(raman['y_av_sub'][i])
            raman['y_sub_norm'].append(raman['y_av_sub_norm'][i])
            raman['y_sub_av'].append(raman['y_av_sub'][i])
            raman['y_sub_std'].append(raman['y_std_sub'][i])
            raman['y_sub_av_norm'].append(raman['y_av_sub_norm'][i])
            raman['y_sub_std_norm'].append(raman['y_std_sub_norm'][i])
    
    # check arrays are still the same length
    for key in ['y_av_sub', 'y_sub_av']:
        if len(raman[key][i]) != len(raman['raman_shift'][i]):
            print
            print("    length of %s array does not match raman shift array for %s!" % (key, raman['ID'][i]))
        
x_start, x_end = (400, 4000)

# plot each baselined spectrum
for i in range(0, len(raman['ID'])):
    plt.figure(figsize=(8,4))
    title = "%s Baselined Spectrum\n%s nm %.2f%% %0.fx%0.1f sec" % (raman['sample'][i], raman['laser_wavelength'][i], raman['laser_power'][i], raman['accumulations'][i], raman['exposure_time'][i])
    if raman['points'][i] > 1:
        title += " %s point average" % raman['points'][i]
    plt.title(title)
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity (counts)")
    plt.plot(raman['raman_shift'][i], raman['y_av_sub'][i], c=Laser_colors[All_Lasers.index(laser)])
    plt.savefig("%s%s_av_sub.png" % (raman['fig_dir'][i], raman['ID'][i]), dpi=300)
    plt.show()


# plot all baselined spectra for each sample that cover x_start - x_end range
x_start, x_end = (None, None)   # set to (None, None) to show all possible ranges
offset = 0.5                    # offset in y axis between spectra
for sample in Sample_IDs:
    # count how many spectra match this sample and range (for determining plot size)
    if x_start != None and x_end != None:
        result = np.ravel(np.where((raman['sample'] == sample) & (x_start >= raman['x_start']) & (raman['x_end'] >= x_end)))
    else:
        result = np.ravel(np.where(raman['sample'] == sample))
    plt.figure(figsize=(8,3+0.5*len(result)))
    plt.title("%s\nBaselined Spectra by Wavelength" % (sample))
    labels = []
    count = 0
    for laser in Lasers:
        # count how many spectra match this sample and wavelength and range
        if x_start != None and x_end != None:
            result = indices[np.ravel(np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser) & (x_start >= raman['x_start']) & (raman['x_end'] >= x_end)))]
        else:
            result = indices[np.ravel(np.where((raman['sample'] == sample) & (raman['laser_wavelength'] == laser)))]
        if len(result) > 0:
            for i in result:
                # plot baselined spectrum
                if laser not in labels:
                    plt.plot(raman['raman_shift'][i], raman['y_av_sub_norm'][i]+count*offset, c=Laser_colors[All_Lasers.index(laser)], label="%s nm" % laser)
                    labels.append(laser)
                else:
                    plt.plot(raman['raman_shift'][i], raman['y_av_sub_norm'][i]+count*offset, c=Laser_colors[All_Lasers.index(laser)])
                count += 1
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Normalised Intensity (offset)")
    plt.yticks([])
    plt.ylim(-0.5*offset, 0.5+count*offset)
    if x_start != None and x_end != None:
        plt.xlim(x_start, x_end)
    plt.legend()
    plt.savefig("%sby sample/%s/%s_sub-spectra.png" % (Fig_dir, sample, sample), dpi=300)
    plt.show()

"""
# ==================================================
# save processed data to files
# ==================================================
"""

debug = False

print()
print("saving processed spectra...")

# each save file contains all spectra for a given sample measured using the same settings

count = 0
for i in range(0, len(raman['ID'])):
    if raman['points'][i] > 1 and Baseline_individual_spectra == True:
        save_data = np.vstack((raman['wavelength'][i], raman['raman_shift'][i], raman['y_av'][i], raman['y_std'][i], raman['y_av_norm'][i], raman['y_std_norm'][i], raman['y_sub_av'][i], raman['y_sub_std'][i], raman['y_sub_av_norm'][i], raman['y_sub_std_norm'][i]))
        header = ['wavelength (nm)', 'raman shift (cm-1)', 'Averaged Raw Intensity (counts)', 'St.Dev. of Raw Intensity (counts)', 'Normalised Averaged Raw Intensity (counts)', 'St.Dev. of Normalised Raw Intensity (counts)','Averaged Baselined Intensity (counts)', 'St.Dev. of Baselined Intensity (counts)', 'Normalised Averaged Baselined Intensity (counts)', 'St.Dev. of Normalised Baselined Intensity (counts)']
    else:
        save_data = np.vstack((raman['wavelength'][i], raman['raman_shift'][i], raman['y_av'][i], raman['y_std'][i], raman['y_av_norm'][i], raman['y_std_norm'][i], raman['y_av_sub'][i], raman['y_std_sub'][i], raman['y_av_sub_norm'][i], raman['y_std_sub_norm'][i]))
        header = ['wavelength (nm)', 'raman shift (cm-1)', 'Averaged Raw Intensity (counts)', 'St.Dev. of Raw Intensity (counts)', 'Normalised Averaged Raw Intensity (counts)', 'St.Dev. of Normalised Raw Intensity (counts)','Averaged Baselined Intensity (counts)', 'St.Dev. of Baselined Intensity (counts)', 'Normalised Averaged Baselined Intensity (counts)', 'St.Dev. of Normalised Baselined Intensity (counts)']
    # add individual point spectra
    for i2 in range(0, raman['points'][i]):
        save_data = np.vstack((save_data, raman['y'][i][i2]))
        header.append("Raw Spectrum Point %0.f" % (i2+1))
    save_name = "%s_%s_%0.f-%0.fcm_%s-point" % (raman['measurement_date'][i].strftime("%Y-%m-%d"), raman['ID'][i], raman['x_start'][i], raman['x_end'][i], raman['points'][i])
    # save data to output folder
    np.savetxt("%s%snm/by sample/%s/%s_spectra.csv" % (Out_dir, raman['laser_wavelength'][i], raman['sample'][i], save_name), save_data.transpose(), header=", ".join(header), delimiter=', ')
    count += 1
print("    %0.f spectra saved to file" % count)

print()
print()
print("total spectra processed:", len(raman['ID']))
for laser in Lasers:
    print("    %4s nm:" % laser, len(raman['ID'][np.where(raman['laser_wavelength'] == laser)]))
print("total samples processed:", len(np.unique(raman['sample'])))
for laser in Lasers:
    print("    %4s nm:" % laser, len(np.unique(raman['sample'][np.where(raman['laser_wavelength'] == laser)])))
print()
print("DONE")

