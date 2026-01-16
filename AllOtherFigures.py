# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:04:58 2025

@author: nlancaster
"""

# Figure 1: prepare 1D plots with a max width of 3.33 in (max depth is 9.167 in)
import pandas as pd

pep2_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep2_FixedExtraZpoint.xlsx")
pep3_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep3_FixedExtraZpoint.xlsx")

pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'


x_data = [pep2_1d['x_rel'][1:14],pep2_1d['Intensity'][1:14],pep3_1d['x_rel'][1:14],pep3_1d['Intensity'][1:14]]
y_data = [pep2_1d['y_rel'][15:32],pep2_1d['Intensity'][15:32],pep3_1d['y_rel'][15:32],pep3_1d['Intensity'][15:32]]
z_data = [pep2_1d['z_rel'][32:45],pep2_1d['Intensity'][32:45],pep3_1d['z_rel'][32:45],pep3_1d['Intensity'][32:45]]


import matplotlib.pyplot as plt
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize = (3.3, 5.5))

ax1.axis('off')

ax2.plot(x_data[0],x_data[1],label = pep2_label,marker = 'o',color = 'royalblue',markersize = 6)
ax2.plot(x_data[2],x_data[3],label = pep3_label,marker = 'o',color = 'maroon',markersize = 6)
ax2.set_xlabel('x position (mm)',fontsize = 8)
ax2.set_xlim(-4,4) #same as y position

ax2.axvline(-0.3,linestyle = '--',color = 'black',linewidth = 1)
ax2.axvline(0.3,linestyle = '--',color = 'black',linewidth = 1)

ax3.plot(y_data[0],y_data[1],label = pep2_label,marker = 'o',color = 'royalblue',markersize = 6)
ax3.plot(y_data[2],y_data[3],label = pep3_label,marker = 'o',color = 'maroon',markersize = 6)
ax3.set_xlabel('y position (mm)',fontsize = 8)
ax3.axvline(-0.8,linestyle = '--',color = 'black',linewidth = 1)
ax3.axvline(0.8,linestyle = '--',color = 'black',linewidth = 1)
ax3.set_ylim(-4,4)

#added 0.856 mm to the z relative position values
ax4.plot([x - 0.856 for x in z_data[0]],z_data[1],label = pep2_label,marker = 'o',color = 'royalblue',markersize = 6)
ax4.plot([x - 0.856 for x in z_data[2]],z_data[3],label = pep3_label,marker = 'o',color = 'maroon',markersize = 6)
ax4.set_xlabel('z position (mm)',fontsize = 8)
ax4.set_xlim(0,-7)
ax4.legend(frameon = False, fontsize = 8,loc = [0.4,0.72])



axes = (ax2,ax3,ax4)

for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params('both',labelsize = 8)
    ax.set_ylabel('Intensity',fontsize = 8)
    ax.yaxis.offsetText.set_fontsize(8)

    ax.set_ylim(0,7e8)
    
    



plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.tight_layout()
# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/ManuscriptPrep/PythonFigures/20251113_EmitterPositioning_1Dplots.svg")


#%%Figure 2: prepare 2D plots with a max width of 3.33 in (max depth is 9.167 in)
import pandas as pd
import numpy as np
#read in dataset
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z3.xlsx')

#remove QC data for heatmap
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

#create table
xtick_labels = [-2,-1,0,1,2]
ytick_labels = [-3,-2,-1,0,1,2,3]

#create dataframe with formatting for heatmap
df_z0 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z0['Intensity']):
    df_z0.at[data_z0['y_rel'][i],data_z0['x_rel'][i]] = val

df_z2 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z2['Intensity']):
    df_z2.at[data_z2['y_rel'][i],data_z2['x_rel'][i]] = val

df_z4 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z4['Intensity']):
    df_z4.at[data_z4['y_rel'][i],data_z4['x_rel'][i]] = val


#%% create plots
import matplotlib.pyplot as plt
import seaborn as sb

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = ( 3.3,6))

vmax = 	710084506.828282

vmin = 0
cmap_name= 'mako'
ax1 = sb.heatmap(df_z0,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax1,vmin = vmin, vmax = vmax,cmap = cmap_name)
ax2 = sb.heatmap(df_z2,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax2,vmin = vmin, vmax = vmax,cmap = cmap_name)
ax3 = sb.heatmap(df_z4,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax3,vmin = vmin, vmax = vmax,cmap = cmap_name)



axes = (ax1, ax2, ax3)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('x (mm)',fontsize = 8)
    ax.set_ylabel('y (mm)',fontsize = 8)
    ax.collections[0].colorbar.set_label("Intensity",fontsize = 8)
    ax.collections[0].colorbar.ax.tick_params(labelsize= 8)
    ax.collections[0].colorbar.ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.invert_yaxis()
    ax.invert_xaxis()
    
ax1.set_title('z = -0.9 mm',fontsize = 8)
ax2.set_title('z = -2.9 mm',fontsize = 8)
ax3.set_title('z = -4.9 mm',fontsize = 8)


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.tight_layout()

fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20260113_EmitterPositioning_HeatMap_Peptide+2_SmallVer.svg")

#%% Example mass spectra for infusion - Figure S3
import pandas as pd
spectra = pd.read_csv('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/20250129_20uLBSA_1Dexp_Exp33_AveragedSpectrum_RemovedHeader.csv')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (5,2.5),layout = 'constrained')
markerline, stemlines, baseline = ax.stem(spectra['m/z'],spectra['Intensity'],markerfmt=" ",basefmt=" ",linefmt='black')



plt.setp(stemlines, 'linewidth', 1)



ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0,)
ax.set_xlim(290,1500)
ax.set_xlabel('m/z')
ax.set_ylabel('Intensity')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20250108_MS_spectra_v1.svg")

#%%Figure S4: prepare supplemental plots showing tic stability over infusion period
from pymsfilereader import MSFileReader


def int_val(raw_file,scan_num,mz,ppm_tol):
    mz_lim = (mz-((ppm_tol*mz)/1e6),mz+((ppm_tol*mz)/1e6))
    
    mass_list = raw_file.GetMassListFromScanNum(scan_num)[0]
    mass_indices = []
    #this part might be a bit slow but I'm just going to do it with a for loop
    for i,x in enumerate(mass_list[0]):
        if (x>=mz_lim[0]) & (x<=mz_lim[1]):
            mass_indices.append(i)
    #add spectra intensities
    spectra_int = []
    for spectra in mass_indices:
        spectra_int.append(mass_list[1][spectra])
    

    return sum(spectra_int)

stability_test = MSFileReader('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/RawFiles/20250129_22uLBSA_StabilityTest01.raw')


pep2_mz = 740.40185
pep3_mz = 480.6093
rt = []
pep2 = []
pep3 = []

mass_tol = 15 #ppm

total_scans = stability_test.NumSpectra

for x in range(1,total_scans):
    rt.append(stability_test.RTFromScanNum(x))
    pep2.append(int_val(stability_test,x,pep2_mz,15))
    pep3.append(int_val(stability_test,x,pep3_mz,15))
#%%
import numpy as np
rt_rsd = []
pep2_rsd = []
pep3_rsd = []

for i,x in enumerate(rt):
    if i+20 == len(rt)-1: #average of 20 scans at a time
        break
    rt_rsd.append(x)
    pep2_rsd.append(np.std(pep2[i:i+20])/np.average(pep2[i:i+20])*100)
    pep3_rsd.append(np.std(pep3[i:i+20])/np.average(pep3[i:i+20])*100)    
#%%
import matplotlib.pyplot as plt

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize = (7,7))



ax1.plot(rt,pep2,color = 'royalblue')
ax1.set_xlabel('Time (min)',fontsize = 12)
ax1.set_ylabel('Intensity',fontsize = 12)
ax1.set_xlim(0,)
ax1.set_ylim(0,)
ax1.set_title('LGEYGFQNALIVR, +2')

ax1.axvline(15,linestyle = '--',color = 'black',linewidth = 1)
ax1.axvline(45,linestyle = '--',color = 'black',linewidth = 1)

ax2.plot(rt_rsd,pep2_rsd,color = 'royalblue')
ax2.set_title('LGEYGFQNALIVR, +2')
ax2.set_xlabel('Time (min)',fontsize = 12)
ax2.set_ylabel('% RSD',fontsize = 12)
ax2.set_ylim(0,15)
ax2.set_xlim(15,45) #show over typical data collection range



ax3.plot(rt,pep3,color = 'maroon')
ax3.set_xlabel('Time (min)',fontsize = 12)
ax3.set_ylabel('Intensity',fontsize = 12)
ax3.set_xlim(0,)
ax3.set_ylim(0,)
ax3.set_title('RHPEYAVSVLLR, +3')

ax3.axvline(15,linestyle = '--',color = 'black',linewidth = 1)
ax3.axvline(45,linestyle = '--',color = 'black',linewidth = 1)

ax4.plot(rt_rsd,pep3_rsd,color = 'maroon')
ax4.set_title('RHPEYAVSVLLR, +3')
ax4.set_xlabel('Time (min)',fontsize = 12)
ax4.set_ylabel('% RSD',fontsize = 12)
ax4.set_ylim(0,15)
ax4.set_xlim(15,45) #show over typical data collection range




axes = (ax1,ax2,ax3,ax4)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params('both',labelsize = 12)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.tight_layout()

# fig.savefig('C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/ManuscriptPrep/PythonFigures/20250202_StablityTest_BSA_Infusion.svg')

#%%Figure S5: prepare supplemental plots showing QC reads for 1D and 2D experiments heatmap generation
import pandas as pd

def rsd_calc(values):
    import numpy as np
    return np.std(values)/np.average(values)*100
    
def qc_data(path):
    import pandas as pd
    df = pd.read_excel(path)
    df = df[df['QC?']== 'QC'].reset_index(drop = True)
    
    return ([*range(1,len(df['Intensity'])+1)],df['Intensity'])

qc_1d_2 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep2.xlsx")
qc_1d_3 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep3.xlsx")
qc_2d_z1_2 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z1.xlsx")
qc_2d_z1_3 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z1.xlsx")
qc_2d_z2_2 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z2.xlsx")
qc_2d_z2_3 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z2.xlsx")
qc_2d_z3_2 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z3.xlsx")
qc_2d_z3_3 = qc_data("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z3.xlsx")


pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'

import matplotlib.pyplot as plt
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (7,6) )

ax1.plot(qc_1d_2[0],qc_1d_2[1],label = pep2_label,marker = 'o',color = 'royalblue')
ax1.plot(qc_1d_3[0],qc_1d_3[1],label = pep3_label,marker = 'o',color = 'maroon')
ax1.set_title('1D',fontsize = 12,fontweight = 'bold')

pep2_rsd = rsd_calc(qc_1d_2[1])
pep3_rsd = rsd_calc(qc_1d_3[1])
x_text_pos = 0.7
y_text_pos = 1
ax1.annotate(str(round(pep2_rsd,1))+'% RSD',(x_text_pos,y_text_pos),xycoords = 'axes fraction',color = 'royalblue',fontsize = 11,fontweight = 'bold',va = 'bottom')
ax1.annotate(str(round(pep3_rsd,1))+'% RSD',(x_text_pos,y_text_pos - 0.09),xycoords = 'axes fraction',color = 'maroon',fontsize = 11,fontweight = 'bold',va = 'bottom')

ax2.plot(qc_2d_z1_2[0],qc_2d_z1_2[1],label = pep2_label,marker = 'o',color = 'royalblue')
ax2.plot(qc_2d_z1_3[0],qc_2d_z1_3[1],label = pep3_label,marker = 'o',color = 'maroon')
ax2.set_title('2D, z = -0.9',fontsize = 12,fontweight = 'bold')


pep2_rsd = rsd_calc(qc_2d_z1_2[1])
pep3_rsd = rsd_calc(qc_2d_z1_3[1])
x_text_pos = 0.7
y_text_pos = 0.4
ax2.annotate(str(round(pep2_rsd,1))+'% RSD',(x_text_pos,y_text_pos),xycoords = 'axes fraction',color = 'royalblue',fontsize = 11,fontweight = 'bold',va = 'bottom')
ax2.annotate(str(round(pep3_rsd,1))+'% RSD',(x_text_pos,y_text_pos - 0.09),xycoords = 'axes fraction',color = 'maroon',fontsize = 11,fontweight = 'bold',va = 'bottom')



ax3.plot(qc_2d_z2_2[0],qc_2d_z2_2[1],label = pep2_label,marker = 'o',color = 'royalblue')
ax3.plot(qc_2d_z2_3[0],qc_2d_z2_3[1],label = pep3_label,marker = 'o',color = 'maroon')
ax3.set_title('2D, z = -2.9',fontsize = 12,fontweight = 'bold')

pep2_rsd = rsd_calc(qc_2d_z2_2[1])
pep3_rsd = rsd_calc(qc_2d_z2_3[1])
x_text_pos = 0.7
y_text_pos = 1
ax3.annotate(str(round(pep2_rsd,1))+'% RSD',(x_text_pos,y_text_pos),xycoords = 'axes fraction',color = 'royalblue',fontsize = 11,fontweight = 'bold',va = 'bottom')
ax3.annotate(str(round(pep3_rsd,1))+'% RSD',(x_text_pos,y_text_pos - 0.09),xycoords = 'axes fraction',color = 'maroon',fontsize = 11,fontweight = 'bold',va = 'bottom')





ax4.plot(qc_2d_z3_2[0],qc_2d_z3_2[1],label = pep2_label,marker = 'o',color = 'royalblue')
ax4.plot(qc_2d_z3_3[0],qc_2d_z3_3[1],label = pep3_label,marker = 'o',color = 'maroon')
ax4.set_title('2D, z = -4.9',fontsize = 12,fontweight = 'bold')
pep2_rsd = rsd_calc(qc_2d_z3_2[1])
pep3_rsd = rsd_calc(qc_2d_z3_3[1])
x_text_pos = 0.7
y_text_pos = 1
ax4.annotate(str(round(pep2_rsd,1))+'% RSD',(x_text_pos,y_text_pos),xycoords = 'axes fraction',color = 'royalblue',fontsize = 11,fontweight = 'bold',va = 'bottom')
ax4.annotate(str(round(pep3_rsd,1))+'% RSD',(x_text_pos,y_text_pos - 0.09),xycoords = 'axes fraction',color = 'maroon',fontsize = 11,fontweight = 'bold',va = 'bottom')




axes = (ax1,ax2,ax3,ax4)

for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params('both',labelsize = 12)
    ax.set_xlabel('QC Number',fontsize = 12)
    ax.set_ylabel('Intensity',fontsize = 12)
    ax.legend(frameon = False, fontsize = 12,loc = [0.15,0.0])
    ax.set_ylim(0,8.2e8)
    
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.tight_layout()

# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20260109_EmitterPositioning_QC_plots_v2.svg")

#%% fitting 1d data to determine the center - Figure S7
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def find_center(df,ax):

    # Convert to numpy arrays
    x = np.array(df['rel pos'])
    y = np.array(df['Intensity'])
    

    peak_max = max(y)
    
    #calculate peak centroid
    def centroid_calc(x,y):

        num = 0
        denom = 0
        for i,val in enumerate(x):
            num += val * y[i]
            denom += y[i]
        centroid = num/denom
        return centroid

    
    
    center = centroid_calc(x,y)
    ax.spines[['top','right']].set_visible(False)



    ax.plot(df['rel pos'],df['Intensity'],color = 'royalblue',marker = 'o')
    ax.set_ylim(0,)
    ax.set_ylabel('Intensity',fontsize = 10)
    ax.axvline( center,linestyle = '--',color = 'black',linewidth = 1.5)
    ax.set_xlim(-4,4)

    ax.tick_params('both',labelsize =10)
    ax.yaxis.get_offset_text().set_fontsize(10)
    ax.annotate(str(round(center,1))+ " mm",(center,peak_max*0.15),rotation = -90,fontsize = 10)
    return center

import pandas as pd
fig,ax = plt.subplots(layout = 'constrained')
pep2_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep2_FixedExtraZpoint.xlsx")
pep3_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep3_FixedExtraZpoint.xlsx")

pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'

import matplotlib.pyplot as plt
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize = (6,6),layout = 'constrained')




x_data = [pep2_1d['x_rel'][1:14],pep2_1d['Intensity'][1:14],pep3_1d['x_rel'][1:14],pep3_1d['Intensity'][1:14]]
x_df = pd.DataFrame({'rel pos':x_data[0],'Intensity':x_data[1]})
find_center(x_df,ax1)
x_df = pd.DataFrame({'rel pos':x_data[2],'Intensity':x_data[3]})
find_center(x_df,ax2)

ax1.set_xlabel('x position (mm)',fontsize = 10)
ax2.set_xlabel('x position (mm)',fontsize = 10)



y_data = [pep2_1d['y_rel'][15:32],pep2_1d['Intensity'][15:32],pep3_1d['y_rel'][15:32],pep3_1d['Intensity'][15:32]]
y_df = pd.DataFrame({'rel pos':y_data[0],'Intensity':y_data[1]})
find_center(y_df,ax3)
y_df = pd.DataFrame({'rel pos':y_data[2],'Intensity':y_data[3]})
find_center(y_df,ax4)

ax3.set_xlabel('y position (mm)',fontsize = 10)
ax4.set_xlabel('y position (mm)',fontsize = 10)


ax1.set_title(pep2_label+ ' (x dim)',fontsize = 10)
ax2.set_title(pep3_label+ ' (x dim)',fontsize = 10)
ax3.set_title(pep2_label+ ' (y dim)',fontsize = 10)
ax4.set_title(pep3_label+ ' (y dim)',fontsize = 10)


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20260112_EmitterPositioning_xyCenterDist_v1.svg")

#%% Figure S8 - Precursor Ratio plot
import pandas as pd

pep2_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep2_FixedExtraZpoint.xlsx")
pep3_1d = pd.read_excel("P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/1D_Data_Pep3_FixedExtraZpoint.xlsx")

pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'



z_data = [pep2_1d['z_rel'][32:45],pep2_1d['Intensity'][32:45],pep3_1d['z_rel'][32:45],pep3_1d['Intensity'][32:45]]


z_data_dif = z_data[1]/z_data[3]

fig,ax = plt.subplots(layout = 'constrained',figsize = (4,3))
ax.spines[['top','right']].set_visible(False)
    

ax.plot(z_data[0],z_data_dif,marker = 'o')


ax.set_ylim(0,1.4)
ax.set_xlim(0.1,-6.1)
ax.set_xlabel('z position (mm)')
ax.set_ylabel('Intensity Ratio')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'


# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20260109_EmitterPositioning_PrecRatio_v1.svg")




#%%Figure S10: prepare 2D plots with a max width of 3.33 in (max depth is 9.167 in)
#for the +3 ion
import pandas as pd
import numpy as np
#read in dataset
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z3.xlsx')

#remove QC data for heatmap
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

#create table
xtick_labels = [-2,-1,0,1,2]
ytick_labels = [-3,-2,-1,0,1,2,3]

#create dataframe with formatting for heatmap
df_z0 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z0['Intensity']):
    df_z0.at[data_z0['y_rel'][i],data_z0['x_rel'][i]] = val

df_z2 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z2['Intensity']):
    df_z2.at[data_z2['y_rel'][i],data_z2['x_rel'][i]] = val

df_z4 = pd.DataFrame(np.zeros((len(ytick_labels),len(xtick_labels))),index = ytick_labels,columns = xtick_labels)
#loop through dataset and define the points
for i,val in enumerate(data_z4['Intensity']):
    df_z4.at[data_z4['y_rel'][i],data_z4['x_rel'][i]] = val



#%% create plots - small version
import matplotlib.pyplot as plt
import seaborn as sb

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = ( 3.3,6))

vmax = 	478579082.787878

vmin = 0
cmap_name= 'mako'
ax1 = sb.heatmap(df_z0,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax1,vmin = vmin, vmax = vmax,cmap = cmap_name)
ax2 = sb.heatmap(df_z2,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax2,vmin = vmin, vmax = vmax,cmap = cmap_name)
ax3 = sb.heatmap(df_z4,xticklabels = xtick_labels,yticklabels=ytick_labels,square = True,ax = ax3,vmin = vmin, vmax = vmax,cmap = cmap_name)



axes = (ax1, ax2, ax3)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('x (mm)',fontsize = 8)
    ax.set_ylabel('y (mm)',fontsize = 8)
    ax.collections[0].colorbar.set_label("Intensity",fontsize = 8)
    ax.collections[0].colorbar.ax.tick_params(labelsize= 8)
    ax.collections[0].colorbar.ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.invert_yaxis()
    ax.invert_xaxis()
    
ax1.set_title('z = -0.9 mm',fontsize = 8)
ax2.set_title('z = -2.9 mm',fontsize = 8)
ax3.set_title('z = -4.9 mm',fontsize = 8)


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

fig.tight_layout()

# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20260113_EmitterPositioning_HeatMap_Peptide+3_SmallVer.svg")




#%% Create Figure S11  - width vs z

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def x_width(df,plot_title, resolution=1000):
    
    df = df[df['y_rel']==0]
    
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
    x = np.array(df['x_rel'])
    y = np.array(df['Intensity'])
    
    # Fit spline
    spline = UnivariateSpline(x, y,s = 0)
    
    # Interpolate
    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = spline(x_fine)
    
    # Find peak and half max
    peak_max = np.max(y_fine)
    half_max = peak_max *0.9
    
    # Find indices where curve crosses half max
    indices = np.where(y_fine >= half_max)[0]
    if len(indices) < 2:
        raise ValueError("Not enough points above 90% max to calculate width.")
    
    width = x_fine[indices[-1]] - x_fine[indices[0]]
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(layout = 'constrained')
    ax.spines[['top','right']].set_visible(False)
    
    ax.scatter(df['x_rel'],df['Intensity'])
    ax.plot(x_fine,y_fine)
    ax.set_ylim(0,)
    ax.set_title(plot_title + f" {width:.3f}")
    ax.axvline( x_fine[indices[0]],linestyle = '--',color = 'black')
    ax.axvline( x_fine[indices[-1]],linestyle = '--',color = 'black')
    plt.show()
    plt.close()
    return width

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def y_width(df,plot_title, resolution=1000):
    
    df = df[df['x_rel']==0]
    
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
    x = np.array(df['y_rel'])
    y = np.array(df['Intensity'])
    
    # Fit spline
    spline = UnivariateSpline(x, y,s = 0)
    
    # Interpolate
    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = spline(x_fine)
    
    # Find peak and half max
    peak_max = np.max(y_fine)
    half_max = peak_max *0.9
    
    # Find indices where curve crosses half max
    indices = np.where(y_fine >= half_max)[0]
    if len(indices) < 2:
        raise ValueError("Not enough points above 90% max to calculate width.")
    
    width = x_fine[indices[-1]] - x_fine[indices[0]]
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(layout = 'constrained')
    ax.spines[['top','right']].set_visible(False)
    
    ax.scatter(df['y_rel'],df['Intensity'])
    ax.plot(x_fine,y_fine)
    ax.set_ylim(0,)
    ax.set_title(plot_title + f" {width:.3f}")
    ax.axvline( x_fine[indices[0]],linestyle = '--',color = 'black')
    ax.axvline( x_fine[indices[-1]],linestyle = '--',color = 'black')
    plt.show()
    plt.close()
    return width






#read in dataset - pep3
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z3.xlsx')
#remove QC data
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

datasets = [data_z0,data_z2,data_z4]


    

pep3_x_width = []
pep3_y_width = []

labels = ['z0','z2','z4']
for i,df in enumerate(datasets):
    pep3_x_width.append(x_width(df,'Pep3, xdim,'+labels[i]))
    pep3_y_width.append(y_width(df,'Pep3, ydim,'+labels[i]))


#read in dataset - pep3
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z3.xlsx')
#remove QC data
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

datasets = [data_z0,data_z2,data_z4]


    

pep2_x_width = []
pep2_y_width = []

labels = ['z0','z2','z4']
for i,df in enumerate(datasets):
    pep2_x_width.append(x_width(df,'Pep2, xdim,'+labels[i]))
    pep2_y_width.append(y_width(df,'Pep2, ydim,'+labels[i]))




#%% make plots

import matplotlib.pyplot as plt
fig,(ax1,ax2) = plt.subplots(2,1,figsize = (5,5),layout = 'constrained')
ax1.spines[['top','right']].set_visible(False)
ax2.spines[['top','right']].set_visible(False)
pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'


z_points = [-0.856,-2.856,-4.856]


ax1.plot(z_points,pep2_x_width,label = pep2_label ,color = 'royalblue',marker = 'x')
ax1.plot(z_points,pep3_x_width,label = pep3_label ,color = 'maroon',marker = 'x')


ax2.plot(z_points,pep2_y_width,label = pep2_label ,color = 'royalblue',marker = 'o')
ax2.plot(z_points,pep3_y_width,label = pep3_label ,color = 'maroon',marker = 'o')

ax1.legend(frameon= False)
ax1.set_xlim(0,-5.2)
ax1.set_ylim(0,3.4)
ax1.set_xlabel('z Position (mm)')
ax1.set_ylabel('Width at 90% Maximum (mm)')

ax2.legend(frameon= False)
ax2.set_xlim(0,-5.2)
ax2.set_ylim(0,3.4)
ax2.set_xlabel('z Position (mm)')
ax2.set_ylabel('Width at 90% Maximum (mm)')


ax1.set_yticks(np.linspace(0,3,7))
ax2.set_yticks(np.linspace(0,3,7))

ax1.set_title('x Dimension')
ax2.set_title('y Dimension')


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'


# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/ManuscriptPrep/PythonFigures/20251107_EmitterPositioning_width_zDependence_v2.svg")
#%% fitting 2d data to determine the center as function of z position - overlaid plots - Figure S12
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def find_center(df,ax,data_label,color, resolution=1000):
    

    
    """
    Estimate center of a peak using spline interpolation.
    
    
    Parameters:
        x (list or array): X-values of the data points.
        y (list or array): Y-values of the data points.
        resolution (int): Number of points for interpolation (default: 1000).
    
    Returns:
        x_fine (array): Interpolated X-values.
        y_fine (array): Interpolated Y-values.
    """
    # Convert to numpy arrays
    x = np.array(df['y_rel'])
    y = np.array(df['Intensity'])
    
    # Fit spline
    spline = UnivariateSpline(x, y,s = 0)
    
    # Interpolate
    x_fine = np.linspace(x.min(), x.max(), resolution)
    y_fine = spline(x_fine)
    


    ax.spines[['top','right']].set_visible(False)

    ax.scatter(df['y_rel'],[val/max(y_fine) for val in df['Intensity']],color = color)
    ax.plot(x_fine,[val/max(y_fine) for val in y_fine],color = color,label = data_label,
            markevery = [-1],marker = 'o')
    ax.set_ylim(0,1.2)
    ax.set_ylabel('Relative Intensity',fontsize = 10)
    # ax.axvline( center,linestyle = '--',color = color,linewidth = 1.5)
    ax.set_xlim(-4.5,4.5)

    ax.tick_params('both',labelsize =10)
    ax.yaxis.get_offset_text().set_fontsize(10)
    # ax.annotate(str(round(center,2))+ " mm",(center,peak_max*0.05),rotation = -90,fontsize = 10)
    ax.set_xlabel('y position (mm)',fontsize = 10)
    
    
    
    

    
    
    return 


import matplotlib.pyplot as plt
fig,(ax1,ax2) = plt.subplots(2,1,figsize = (4,7),layout = 'constrained')
ax1.axvline(0,linewidth = 0.5,color = 'black')
ax2.axvline(0,linewidth = 0.5,color = 'black')
#read in dataset - pep2
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep2_z3.xlsx')

#filter just for x = 0 for y dimension data
data_z0 = data_z0[data_z0['x_rel']==0]
data_z2 = data_z2[data_z2['x_rel']==0]
data_z4 = data_z4[data_z4['x_rel']==0]

#remove QC data
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

find_center(data_z4,ax1, 'z = -4.9','red')


find_center(data_z2,ax1, 'z = -2.9','blue')

find_center(data_z0,ax1, 'z = -0.9','black')


#read in dataset - pep3
data_z0 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z1.xlsx')
data_z2 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z2.xlsx')
data_z4 = pd.read_excel('P:/Projects/NML_2024_Internal_MultiColumn/EmitterInterferenceAssessments/20250117_PositioningExp_Rd03/ProcessingFiles/2DData_ForDataProcessingWorkflowDev_Pep3_z3.xlsx')

#filter just for x = 0 for y dimension data
data_z0 = data_z0[data_z0['x_rel']==0]
data_z2 = data_z2[data_z2['x_rel']==0]
data_z4 = data_z4[data_z4['x_rel']==0]

#remove QC data
data_z0 = data_z0[data_z0['QC?']!='QC'].reset_index(drop = True)
data_z2 = data_z2[data_z2['QC?']!='QC'].reset_index(drop = True)
data_z4 = data_z4[data_z4['QC?']!='QC'].reset_index(drop = True)

find_center(data_z4,ax2, 'z = -4.9','red')


find_center(data_z2,ax2, 'z = -2.9','blue')

find_center(data_z0,ax2, 'z = -0.9','black')




pep2_label = 'LGEYGFQNALIVR, +2'
pep3_label = 'RHPEYAVSVLLR, +3'

legend_loc = [0.64,0.81]



handles,labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1],labels[::-1],frameon = False,loc = 'upper left')
handles,labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1],labels[::-1],frameon = False,loc = 'upper left')

ax1.set_title(pep2_label,fontsize = 10)
ax2.set_title(pep3_label,fontsize = 10)



plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

# fig.savefig("C:/Users/nlancaster/OneDrive - UW-Madison/2024_EmitterPosition_JASMS_Note_Manuscript/JASMS_Revision_Rd01/PythonFigures/20251208_zDep_Ypos_Overlay_v1.svg")
