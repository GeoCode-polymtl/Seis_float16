#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the results from inversion
"""

import matplotlib as mpl
import scipy.ndimage as image
import h5py as h5
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Read the modeling constants
"""
csts = h5.File('marmfp32_csts.mat','r')
nab = int(csts['nab'][0])
NZ = csts['N'][0]-2*nab
NX = csts['N'][1]-2*nab
dh = int(csts['dh'][0])
csts.close()

"""
Read the models
"""
file = h5.File('marmfp32_model.mat','r')
model_true = np.transpose(file['vp'][nab:-nab, nab:-nab] / 1000)
file.close()
file = h5.File('marminv_fp32logs/freq1.2/results.mat','r')
model_init = (file['vp'][0, nab:-nab, nab:-nab] / 1000)
file.close()
file = h5.File('marminv_fp32logs/freq17/results.mat','r')
model_inv32 = (file['vp'][-1, nab:-nab, nab:-nab] / 1000)
file.close()
file = h5.File('marminv_fp16iologs/freq17/results.mat','r')
model_inv16io = (file['vp'][-1, nab:-nab, nab:-nab] / 1000)
file.close()
file = h5.File('marminv_fp16comlogs/freq17/results.mat','r')
model_inv16comp = (file['vp'][-1, nab:-nab, nab:-nab] / 1000)
file.close()

fig = plt.figure(figsize=(18/2.54, 10/2.54))
gridspec.GridSpec(2,2)

axlabels = ['a)', 'b)', 'c)', 'd)']

def plot_marm(ind=0, model=None):

    plot_ref = True
    if model is None:
        model = model_true
        plot_ref = False
    ax = plt.subplot2grid( (2,2), (int(ind/2),int(ind%2)), colspan=1, rowspan=1)
    im = ax.imshow(model, interpolation='bilinear', vmax=6, vmin=1.5,
                   cmap=plt.get_cmap('jet'),
                   extent=[0, (NX + 1) * dh / 1000, (NZ + 1) * dh / 1000, 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.06)

    clr = plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    clr.set_ticks(np.arange(1.5, 6.1, 0.75))
    cax.xaxis.tick_top()
    cax.set_xlabel('V (km/s)', labelpad=4)
    cax.xaxis.set_label_position('top')

    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 4))
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.text(-1.2,-1, axlabels[ind])


    ax.set_aspect(1)


    ax1 = divider.append_axes("right", size="7%", pad=0.1)
    ax1.plot(model_true[:, int(7000 / dh)].flatten(),
             (np.arange(0, (NZ)) * dh / 1000).flatten(), linewidth=1.0,
             label='True')
    if plot_ref:
        ax1.plot(model[:, int(7000 / dh)].flatten(),
                 (np.arange(0, (NZ)) * dh / 1000).flatten(), linewidth=1.0,
                 label='True')
    ax1.invert_yaxis()
    ax1.set_xlim(left=1.2, right=6)
    ax1.set_xticks([1.5, 6])

    ax1.set_xlabel('V(km/s)')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_yticks([])

    ax.annotate('', xy=(7, 4), xycoords='axes fraction', xytext=(10, 4),
            arrowprops=dict(arrowstyle="<->", color='b'))

    #ax1.set_yticks(np.arange(0, 4))

#plt.tight_layout()


plot_marm(ind=2, model=model_inv32)
plot_marm(ind=3, model=model_inv16comp)
plot_marm(ind=0)
plot_marm(ind=1, model=model_init)
plt.tight_layout()
plt.savefig('marm_inv.png', dpi=600)
plt.savefig('marm_inv_lowres.png', dpi=100)
plt.show()

print(np.sqrt(np.sum( (model_inv32-model_true)**2)/model_inv32.flatten().shape[0])*1000)
print(np.sqrt(np.sum( (model_inv32-model_inv16io)**2)/model_inv32.flatten().shape[0])*1000)
print(np.sqrt(np.sum( (model_inv32-model_inv16comp)**2)/model_inv32.flatten().shape[0])*1000)


