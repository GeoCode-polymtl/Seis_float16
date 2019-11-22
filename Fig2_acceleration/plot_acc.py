#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import scipy.ndimage as image
import h5py as h5
#mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})


dims = ['2D','3D']
models = ['k40','m40','p100','v100']
cases = ['FP16 IO', 'FP16 COMP']
modelcases = {'k40': [1,2],'m40': [1,2],'p100': [1,2,3],'v100': [1,2,3]}
casecodes = {'FP16 IO': 2, 'FP16 COMP': 3}


# These are the "Tableau 20" colors as RGB.
colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r / 255., g / 255., b / 255.)


"""
___________________________Acceleration figure__________________________________
"""
fig, ax = plt.subplots(1,1, figsize=(9/2.54, 10/2.54))
bars=[None]*len(dims)*len(cases)
labels=[None]*len(dims)*len(cases)
width = 0.75/(len(dims))
ind = np.arange(0,len(models))
for (ii,dim) in enumerate(dims):
    for (jj,case) in enumerate(cases[::-1]):
        thisa = [float('Nan')]*len(models)
        thise = [float('Nan')]*len(models)
        for (kk,model) in enumerate(models):
            data = h5.File('times_'+dim+'_'+model+'.mat','r')
            if casecodes[case] in modelcases[model]:
                nstr = str(casecodes[case])
                d0 = data['timesFP1'][24:-6]
                dc = data['timesFP'+nstr][24:-6]
                thisa[kk] = np.mean(d0/dc)
                thise[kk] = np.std(d0/dc)
            data.close()
            bars[ii*len(dims)+jj] = ax.bar(ind+(ii)*width, thisa,
                                           width, yerr=thise,
                                           color = colors[ii*len(dims)+jj],
                                           edgecolor="none", zorder=1,
                                           error_kw=dict(ecolor='gray', lw=1,
                                                         capsize=4, capthick=1))
            labels[ii*len(dims)+jj]=dim+': '+case

# add some text for labels, title and axes ticks
ax.set_ylabel('Acceleration', labelpad=5)
ax.set_xlabel('Model', labelpad=5)
#ax.legend(bars, labels, frameon=False, bbox_to_anchor=(1, 1.13))
ax.legend(bars, labels, loc='upper left', bbox_to_anchor=(0.55, 1.12))


# Beautify the figure
ax.set_xticks(ind + width *len(cases)/ 2)
ax.set_xticklabels(models)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
ax.set_xlim([-0.5*width,len(models)])
ax.set_yticks(np.arange(0,2.5,0.5))
ax.set_ylim([0,2.3])
for y in np.arange(0.5, 2.5, 0.5):
    plt.plot([-0.5*width,len(models)],
             [y] * 2,
             "--",
             lw=0.5, color="lightgray", zorder=0)
plt.plot([-0.49*width,-0.49*width], [0,2.3], lw=1, color="lightgray", zorder=0)
plt.plot([-0.49*width,len(models)], [0,0], lw=1, color="lightgray", zorder=0)
#plt.tight_layout()
plt.savefig('accelaration.eps',dpi=300)


"""
___________________________Time for update figure_______________________________
"""
sizes = {'2D':np.arange(512,8192, 64), '3D':np.arange(64,600, 8)}
cases = ['FP32', 'FP16 IO', 'FP16 COMP']
casecodes = {'FP32': 0,'FP16 IO': 2, 'FP16 COMP': 4}

# These are the "Tableau 20" colors as RGB.
colors = [(81,86,143), (71,113,165), (150,177,210),
          (229, 83, 0), (255, 127, 14), (255, 187, 120) ]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r / 255., g / 255., b / 255.)


fig, ax = plt.subplots(1, 1, figsize=(9/2.54,10/2.54))
bars=[None]*len(dims)*len(cases)
labels=[None]*len(dims)*len(cases)
width = 0.75/(len(dims))
ind = np.arange(0,len(models))

data = h5.File('times_'+dim+'_'+model+'.mat','r')


for (ii,dim) in enumerate(dims):
    for (jj,case) in enumerate(cases[::-1]):
        thisa = [float('Nan')]*len(models)
        thise = [float('Nan')]*len(models)
        for (kk,model) in enumerate(models):
            data = h5.File('times_'+dim+'_'+model+'.mat','r')
            if casecodes[case] in modelcases[model]:
                nstr = str(casecodes[case])
                d = data['timesFP'+nstr][24:-6]
                size = sizes[dim][24:-6]
                if dim =='2D':
                    d = [d[el] / size[el] ** 2 for el in range(0, len(size))]
                else:
                    d = [d[el] / size[el] ** 3 for el in range(0, len(size))]
                d=1.0/np.array(d)*10**-6
                thisa[kk] = np.mean(d)
                thise[kk] = np.std(d)

            data.close()
            bars[ii*len(cases)+jj] = ax.bar(ind+(ii)*width, thisa,
                                           width, yerr=thise,
                                           color = colors[ii*len(cases)+jj],
                                           edgecolor="none", zorder=1,
                                           error_kw=dict(ecolor='gray', lw=1,
                                                         capsize=4, capthick=1))
            labels[ii*len(cases)+jj]=dim+': '+case

# add some text for labels, title and axes ticks
ax.set_ylabel('Gridpoint per second x 10$^9$', labelpad=10)
ax.set_xlabel('Model', labelpad=5)
legend = ax.legend(bars, labels, frameon=False, loc='upper left', shadow=True)
legend.get_frame().set_facecolor('w')

# Beautify the figure
ax.set_xticks(ind + width *len(dims)/ 2)
ax.set_xticklabels(models)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
ax.set_xlim([-0.5*width,len(models)])
ax.set_yticks(np.arange(0,6,1))
#ax.set_ylim([0,2.3])
for y in np.arange(0, 6, 1):
    plt.plot([-0.5*width,len(models)],
             [y] * 2,
             "--",
             lw=0.5, color="lightgray", zorder=0)
plt.plot([-0.49*width,-0.49*width], [0,6], lw=1, color="lightgray", zorder=0)
plt.plot([-0.49*width,len(models)], [0,0], lw=1, color="lightgray", zorder=0)


plt.savefig('GPS.eps',dpi=300)