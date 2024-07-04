# Plastic-Raman-analysis
Python scripts for processing and analysing Raman spectra of plastic pollution

These scripts process spectra of plastic debris, generate summary figures, and train classification algorithms to assign spectra to known materials.
  Raman_processing: takes raw Raman spectra and processes them, including baseline subtraction, normalisation, and standardisation of frequency values
  Raman_fitting: takes processed Raman spectra and does mathemetical fitting to determine peak properties
  Raman_analysis: takes processed Raman spectra and fitting results and generates summary figures
  Raman_characterisation: trains machine learning models for classifying materials

Written by Dr Joseph Razzell Hollis on 2024-07-04. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the Journal of Hazardous Materials in 2024 (DOI: TBC). Please cite the methods paper if you adapt this code for your own analysis.

Any updates to this script will be made available online at www.github.com/Jobium/Plastic-Raman-analysis/

Python code requires Python 3.7 (or higher) and the following packages: os, math, glob, datetime, numpy, pandas, matplotlib, lmfit, scipy, skimage, sklearn.
