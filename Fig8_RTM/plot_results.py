#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the results from inversion
"""

import matplotlib as mpl
import scipy.ndimage as image
import h5py as h5
import numpy as np
#mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import laplace
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
dw = 80

"""
Read the models
"""
file = h5.File('marmgrad_fp32_gout.mat','r')
rtm_fp32 = laplace(np.transpose(file['gradvp'][nab:-nab, nab+dw:-nab] / 1000))
file.close()
file = h5.File('marmgrad_fp16io_gout.mat','r')
rtm_fp16io = laplace(np.transpose(file['gradvp'][nab:-nab, nab+dw:-nab] / 1000))
file.close()
file = h5.File('marmgrad_fp16com_gout.mat','r')
rtm_fp16com = laplace(np.transpose(file['gradvp'][nab:-nab, nab+dw:-nab] / 1000))
file.close()

fig = plt.figure(figsize=(18/2.54, 15/2.54))
gridspec.GridSpec(3, 2)

axlabels = ['a)', 'b)', 'c)', 'd)', 'd)', 'e)']
axlabels2 = ['                FP32',
             '             FP16 IO',
             '           FP16 COMP',
             'd)',
             '  (FP32-FP16 IO)X75',
             '(FP32-FP16 COMP)X75']

vmin = np.min(rtm_fp32) * 0.03
vmax = -vmin

def plot_marm(ind=0, model=None):

    ax = plt.subplot2grid( (3, 2), (int(ind%3),int(ind/3)), colspan=1, rowspan=1)
    im = ax.imshow(model, interpolation='bilinear', vmax=vmax, vmin=vmin,
                   cmap=plt.get_cmap('Greys'),
                   extent=[0, (NX + 1) * dh / 1000, (NZ + 1) * dh / 1000, 0])

    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 4))
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.text(-0.3, -0.4, axlabels[ind])
    ax.text(9.3, -0.12, axlabels2[ind], ha='right', weight='bold')
    ax.set_aspect(1)

plot_marm(ind=0, model=rtm_fp32)
plot_marm(ind=1, model=rtm_fp16io)
plot_marm(ind=2, model=rtm_fp16com)
plot_marm(ind=4, model=(rtm_fp32-rtm_fp16io)*75)
plot_marm(ind=5, model=(rtm_fp32-rtm_fp16com)*75)
plt.tight_layout()
plt.savefig('marm_rtm.png', dpi=600)
plt.savefig('marm_rtm_lowres.png', dpi=100)
plt.show()

err_fp16io = 100*(np.sum((rtm_fp32-rtm_fp16io)**2)/np.sum((rtm_fp32)**2))
err_fp16com = 100*(np.sum((rtm_fp32-rtm_fp16com)**2)/np.sum((rtm_fp32)**2))
#err_fp16io = np.median(np.abs(rtm_fp32-rtm_fp16io)/np.std(rtm_fp32))
#err_fp16com = np.median(np.abs(rtm_fp32-rtm_fp16com)/np.std(rtm_fp32))

print("The error between FP32 and FP16 IO is %f" % err_fp16io)
print("The error between FP32 and FP16 COMP is %f" % err_fp16com)

