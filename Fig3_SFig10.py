# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:07:41 2025

@author: nlancaster
"""


import pandas as pd
#read in data from the nearest z and farthest z points
z_data_near = pd.read_csv('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/20250129_20uLBSA_1Dexp_Exp33_AveragedSpectrum_RemovedHeader.csv')
z_data_far = pd.read_csv('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/20250129_20uLBSA_1Dexp_Exp46_AveragedSpectrum_RemovedHeader.csv')

# filter far m/z dataset for detected charge states
z_data_far = z_data_far[z_data_far['Charge']>1]
# filter far m/z dataset for S/N > 50
z_data_far = z_data_far[z_data_far['Signal To Noise']>=300]

# the max number of matches was one, so I don't need to worry about multiple peak assignments
far_match_mz = [] #these are the m/z peaks detected in the 'near' data set that match to peaks in the 

#match peaks that are within 5ppm of each other
ppm_tol = 5
for peak in z_data_far['m/z']:
    peak_tol = (peak-((ppm_tol*peak)/1e6),peak+((ppm_tol*peak)/1e6))# add ppm range
    #match peaks
    match_found = False
    for test_peak in z_data_near['m/z']:
        if test_peak >=peak_tol[0] and test_peak<=peak_tol[1]:
            match_found = True
            far_match_mz.append(test_peak)
            break #exit the for loop since match is found
    if match_found == False:
        far_match_mz.append('None')


            
z_data_far['Near m/z Match'] = far_match_mz

#Filter out no matches
z_data_far = z_data_far[z_data_far['Near m/z Match']!= 'None']

#create a dictionary that returns the intensities of peaks in the near dataset
near_dict = {}
for i,peak in enumerate(z_data_near['m/z']):
    near_dict[peak] = z_data_near['Intensity'][i]

#loop through the far dataset and assign the near intensities
near_int= []
for mz in z_data_far['Near m/z Match']:
    near_int.append(near_dict[mz])
z_data_far['Near Intensity'] = near_int

#calculate fold change
z_data_far['Fold change'] = z_data_far['Intensity']/z_data_far['Near Intensity']

df_global_xy = z_data_far
#%%

import numpy as np
        

def int_val(raw_file,scan_num,mz,ppm_tol):

    mz_lim = (mz-((ppm_tol*mz)/1e6),mz+((ppm_tol*mz)/1e6))
    mass_list = raw_file.GetMassListFromScanNum(scan_num)[0]
    
    
    mz_array = np.array(mass_list[0])
    intensity_array = np.array(mass_list[1])


    # Apply filter using NumPy boolean mask
    mask = (mz_array >= mz_lim[0]) & (mz_array <= mz_lim[1])
    
    # Filtered arrays

    filtered_intensity = intensity_array[mask]
    return filtered_intensity.sum()






def average_int(raw_file,ppm,mz):



    total_scans = raw_file.NumSpectra
    
    intensities = []
    
    for x in range(1,total_scans):
        intensities.append(int_val(raw_file,x,mz,ppm))

        
    return np.average(intensities)




files = ["P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp01.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp02.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp03.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp04.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp05.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp06.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp07.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp08.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp09.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp10.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp11.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp12.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp13.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp14.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp15.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp16.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp17.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp18.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp19.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp20.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp21.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp22.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp23.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp24.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp25.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp26.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp27.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp28.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp29.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp30.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp31.raw",
"P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_20uLBSA_1Dexp_Exp32.raw"]


total_files = len(files)

import os 
file_names = []
for x in files:
    file_names.append(os.path.basename(x))



import pandas as pd

output_df = pd.DataFrame({'File Name':file_names})

column_names = []
data_results = []

for mz in z_data_far['m/z']:
    column_names.append(mz)
    data_results.append([])

from pymsfilereader import MSFileReader

for i,file in enumerate(files):

    raw_file = MSFileReader(file)
    for q,mz in enumerate(column_names):
        data_results[q].append(average_int(raw_file,5,mz))
        print(q)
    print(i+1,'of ' + str(total_files))


output_dict = {'File Name':file_names}
for i,col in enumerate(column_names):
    output_dict[col] = data_results[i]

output_df = pd.DataFrame(output_dict)

output_df.to_csv("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/GlobalRawProcess_raw_process_5ppm_SN300.csv",index = False)
#%% Main Text Figure 3
output_df = pd.read_csv("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/GlobalRawProcess_raw_process_5ppm_SN300.csv")
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def width_90pcnt(rel_pos,intensities,plot_title, resolution=5000):
    

    
    """
    Estimate width of a peak using spline interpolation.
    
    
    Parameters:
        x (list or array): X-values of the data points.
        y (list or array): Y-values of the data points.
        resolution (int): Number of points for interpolation (default: 1000).
    
    Returns:
        width(float): Estimated Full Width at 90% Maximum.
        x_fine (array): Interpolated X-values.
        y_fine (array): Interpolated Y-values.
    """
    # Convert to numpy arrays
    x = np.array(rel_pos)
    y = np.array(intensities)
    
    # Fit spline
    spline = UnivariateSpline(x, y,s = 0)
    
    # Interpolate
    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = spline(x_fine)
    
    # Find peak and half max
    peak_max = np.max(y_fine)
    half_max = peak_max *0.5
    
    # Find indices where curve crosses half max
    indices = np.where(y_fine >= half_max)[0]
    if len(indices) < 2:
        raise ValueError("Not enough points above 90% max to calculate width.")
    
    width = x_fine[indices[-1]] - x_fine[indices[0]]
    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(layout = 'constrained')
    # ax.spines[['top','right']].set_visible(False)
    
    # ax.scatter(rel_pos,intensities)
    # ax.plot(x_fine,y_fine)
    # ax.set_ylim(0,)
    # ax.set_title(plot_title + f" {width:.3f}")
    # ax.axvline( x_fine[indices[0]],linestyle = '--',color = 'black')
    # ax.axvline( x_fine[indices[-1]],linestyle = '--',color = 'black')
    # plt.show()
    # plt.close()
    
    return width
    return peak_max,width
        






x_files = ['20250129_20uLBSA_1Dexp_Exp02.raw',
'20250129_20uLBSA_1Dexp_Exp03.raw',
'20250129_20uLBSA_1Dexp_Exp04.raw',
'20250129_20uLBSA_1Dexp_Exp05.raw',
'20250129_20uLBSA_1Dexp_Exp06.raw',
'20250129_20uLBSA_1Dexp_Exp07.raw',
'20250129_20uLBSA_1Dexp_Exp08.raw',
'20250129_20uLBSA_1Dexp_Exp09.raw',
'20250129_20uLBSA_1Dexp_Exp10.raw',
'20250129_20uLBSA_1Dexp_Exp11.raw',
'20250129_20uLBSA_1Dexp_Exp12.raw',
'20250129_20uLBSA_1Dexp_Exp13.raw',
'20250129_20uLBSA_1Dexp_Exp14.raw']

y_files = ['20250129_20uLBSA_1Dexp_Exp16.raw',
'20250129_20uLBSA_1Dexp_Exp17.raw',
'20250129_20uLBSA_1Dexp_Exp18.raw',
'20250129_20uLBSA_1Dexp_Exp19.raw',
'20250129_20uLBSA_1Dexp_Exp20.raw',
'20250129_20uLBSA_1Dexp_Exp21.raw',
'20250129_20uLBSA_1Dexp_Exp22.raw',
'20250129_20uLBSA_1Dexp_Exp23.raw',
'20250129_20uLBSA_1Dexp_Exp24.raw',
'20250129_20uLBSA_1Dexp_Exp25.raw',
'20250129_20uLBSA_1Dexp_Exp26.raw',
'20250129_20uLBSA_1Dexp_Exp27.raw',
'20250129_20uLBSA_1Dexp_Exp28.raw',
'20250129_20uLBSA_1Dexp_Exp29.raw',
'20250129_20uLBSA_1Dexp_Exp30.raw',
'20250129_20uLBSA_1Dexp_Exp31.raw',
'20250129_20uLBSA_1Dexp_Exp32.raw']
x_rel = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]

y_rel = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]

x_int = []
x_widths = []

x_df = output_df[output_df['File Name'].isin(x_files)]


for col in x_df.columns:
    if col == 'File Name':
        continue
    else:

        x_widths.append(width_90pcnt(x_rel,x_df[col],str(col)))

y_widths = []

y_df = output_df[output_df['File Name'].isin(y_files)]


for col in y_df.columns:
    if col == 'File Name':
        continue
    else:
        y_widths.append(width_90pcnt(y_rel,y_df[col],str(col)))


#%% 
import pandas as pd
import matplotlib.pyplot as plt


#create a dictionary that returns the intensities of peaks in the near dataset
near_dict = {}
for i,peak in enumerate(z_data_near['m/z']):
    near_dict[peak] = z_data_near['Intensity'][i]

#loop through the far dataset and assign the near intensities
near_int= []
for mz in z_data_far['Near m/z Match']:
    near_int.append(near_dict[mz])
z_data_far['Near Intensity'] = near_int

#calculate fold change
z_data_far['Fold change'] = z_data_far['Intensity']/z_data_far['Near Intensity']

from scipy.stats import pearsonr
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (3.2,5.2),layout = 'constrained')



#x width correlation
ax1.scatter(df_global_xy['m/z'],x_widths,color = 'white',edgecolor = 'black',marker = 'o',s = 10)
#pearson correlation
corr_coeff = pearsonr(df_global_xy['m/z'],x_widths)[0]
# corr_coeff = pearsonr(x_plot_data['m/z'],x_plot_data['x widths'])[0]
ax1.annotate("R = " + str(round(corr_coeff,4)),(1,1),fontsize = 8,xycoords = 'axes fraction',ha = 'right',va = 'top')
ax1.set_xlabel('m/z',fontsize = 8)
ax1.set_ylim(0,3.7)

ax1.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5])
ax1.set_ylabel('x Dimension FWHM (mm)',fontsize = 8)


#y width correlation
ax2.scatter(df_global_xy['m/z'],y_widths,color = 'white',edgecolor = 'black',marker = 'o',s = 10)
#pearson correlation
corr_coeff = pearsonr(df_global_xy['m/z'],y_widths)[0]
ax2.annotate("R = " + str(round(corr_coeff,4)),(1,1),fontsize = 8,xycoords = 'axes fraction',ha = 'right',va = 'top')
ax2.set_xlabel('m/z',fontsize = 8)
ax2.set_ylim(0,3.7)
ax2.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5])
ax2.set_ylabel('y Dimension FWHM (mm)',fontsize = 8)

#m/z correlation
ax3.scatter(z_data_far['m/z'],z_data_far['Fold change'],color = 'white',edgecolor = 'black',marker = 'o',s = 10)
#pearson correlation
corr_coeff = pearsonr(z_data_far['m/z'],z_data_far['Fold change'])[0]
ax3.annotate("R = " + str(round(corr_coeff,4)),(1,1),fontsize = 8,xycoords = 'axes fraction',ha = 'right',va = 'top')
ax3.set_xlabel('m/z',fontsize = 8)

ax3.set_ylabel('Fold Change (z=-6.856 / z=-0.856)',fontsize = 8)
ax3.set_ylim(0,1)


axes = (ax1,ax2,ax3)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(275,1525)

    ax.tick_params(axis='both', which='major', labelsize=8)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/ManuscriptPrep/PythonFigures/20251217_GlobalPositionEffects_v2.svg")
#%% make intensity vs fwhm, fold change plots -Supplemental Figure S10
output_df = pd.read_csv("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/GlobalRawProcess_raw_process_5ppm_SN300.csv")
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def width_90pcnt(rel_pos,intensities,plot_title, resolution=5000):
    

    
    """
    Estimate width of a peak using spline interpolation.
    
    
    Parameters:
        x (list or array): X-values of the data points.
        y (list or array): Y-values of the data points.
        resolution (int): Number of points for interpolation (default: 1000).
    
    Returns:
        width(float): Estimated Full Width at 90% Maximum.
        x_fine (array): Interpolated X-values.
        y_fine (array): Interpolated Y-values.
    """
    # Convert to numpy arrays
    x = np.array(rel_pos)
    y = np.array(intensities)
    
    # Fit spline
    spline = UnivariateSpline(x, y,s = 0)
    
    # Interpolate
    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = spline(x_fine)
    
    # Find peak and half max
    peak_max = np.max(y_fine)
    half_max = peak_max *0.5
    
    # Find indices where curve crosses half max
    indices = np.where(y_fine >= half_max)[0]
    if len(indices) < 2:
        raise ValueError("Not enough points above 90% max to calculate width.")
    
    width = x_fine[indices[-1]] - x_fine[indices[0]]
    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(layout = 'constrained')
    # ax.spines[['top','right']].set_visible(False)
    
    # ax.scatter(rel_pos,intensities)
    # ax.plot(x_fine,y_fine)
    # ax.set_ylim(0,)
    # ax.set_title(plot_title + f" {width:.3f}")
    # ax.axvline( x_fine[indices[0]],linestyle = '--',color = 'black')
    # ax.axvline( x_fine[indices[-1]],linestyle = '--',color = 'black')
    # plt.show()
    # plt.close()
    
    # return width
    return peak_max,width
        





x_files = ['20250129_20uLBSA_1Dexp_Exp02.raw',
'20250129_20uLBSA_1Dexp_Exp03.raw',
'20250129_20uLBSA_1Dexp_Exp04.raw',
'20250129_20uLBSA_1Dexp_Exp05.raw',
'20250129_20uLBSA_1Dexp_Exp06.raw',
'20250129_20uLBSA_1Dexp_Exp07.raw',
'20250129_20uLBSA_1Dexp_Exp08.raw',
'20250129_20uLBSA_1Dexp_Exp09.raw',
'20250129_20uLBSA_1Dexp_Exp10.raw',
'20250129_20uLBSA_1Dexp_Exp11.raw',
'20250129_20uLBSA_1Dexp_Exp12.raw',
'20250129_20uLBSA_1Dexp_Exp13.raw',
'20250129_20uLBSA_1Dexp_Exp14.raw']

y_files = ['20250129_20uLBSA_1Dexp_Exp16.raw',
'20250129_20uLBSA_1Dexp_Exp17.raw',
'20250129_20uLBSA_1Dexp_Exp18.raw',
'20250129_20uLBSA_1Dexp_Exp19.raw',
'20250129_20uLBSA_1Dexp_Exp20.raw',
'20250129_20uLBSA_1Dexp_Exp21.raw',
'20250129_20uLBSA_1Dexp_Exp22.raw',
'20250129_20uLBSA_1Dexp_Exp23.raw',
'20250129_20uLBSA_1Dexp_Exp24.raw',
'20250129_20uLBSA_1Dexp_Exp25.raw',
'20250129_20uLBSA_1Dexp_Exp26.raw',
'20250129_20uLBSA_1Dexp_Exp27.raw',
'20250129_20uLBSA_1Dexp_Exp28.raw',
'20250129_20uLBSA_1Dexp_Exp29.raw',
'20250129_20uLBSA_1Dexp_Exp30.raw',
'20250129_20uLBSA_1Dexp_Exp31.raw',
'20250129_20uLBSA_1Dexp_Exp32.raw']
x_rel = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]

y_rel = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]

x_int = []
x_widths = []

x_df = output_df[output_df['File Name'].isin(x_files)]


for col in x_df.columns:
    if col == 'File Name':
        continue
    else:
        intensity_val, width_val = width_90pcnt(x_rel,x_df[col],str(col))
        x_int.append(intensity_val)
        x_widths.append(width_val)




y_df = output_df[output_df['File Name'].isin(y_files)]


y_int = []
y_widths = []
for col in y_df.columns:
    if col == 'File Name':
        continue
    else:
        intensity_val, width_val = width_90pcnt(y_rel,y_df[col],str(col))
        y_int.append(intensity_val)
        y_widths.append(width_val)

#%%        
import pandas as pd
import matplotlib.pyplot as plt


#create a dictionary that returns the intensities of peaks in the near dataset
near_dict = {}
for i,peak in enumerate(z_data_near['m/z']):
    near_dict[peak] = z_data_near['Intensity'][i]

#loop through the far dataset and assign the near intensities
near_int= []
for mz in z_data_far['Near m/z Match']:
    near_int.append(near_dict[mz])
z_data_far['Near Intensity'] = near_int

#calculate fold change
z_data_far['Fold change'] = z_data_far['Intensity']/z_data_far['Near Intensity']


fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (3.2,5.2),layout = 'constrained')



#x width correlation

ax1.scatter([np.log10(val) for val in x_int],x_widths,color = 'white',edgecolor = 'black',marker = 'o',s = 10)
ax1.set_xlabel('Log10 Peak Intensity',fontsize = 8)
ax1.set_ylim(0,3.7)
ax1.set_ylabel('x Dimension FWHM (mm)',fontsize = 8)
ax1.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5])

#y width correlation
ax2.scatter([np.log10(val) for val in y_int],y_widths,color = 'white',edgecolor = 'black',marker = 'o',s = 10)
ax2.set_xlabel('Log10 Peak Intensity',fontsize = 8)
ax2.set_ylim(0,3.7)
ax2.set_ylabel('y Dimension FWHM (mm)',fontsize = 8)
ax2.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5])

#m/z correlation
ax3.scatter([np.log10(val) for val in z_data_far['Near Intensity']],z_data_far['Fold change'],color = 'white',edgecolor = 'black',marker = 'o',s = 10)

ax3.set_xlabel('Log10 Peak Intensity',fontsize = 8)

ax3.set_ylabel('Fold Change (z=-6.856 / z=-0.856)',fontsize = 8)
ax3.set_ylim(0,1)


axes = (ax1,ax2,ax3)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)


    ax.tick_params(axis='both', which='major', labelsize=8)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/ManuscriptPrep/PythonFigures/20251217_GlobalPositionEffectsVsIntensity_v1.svg")