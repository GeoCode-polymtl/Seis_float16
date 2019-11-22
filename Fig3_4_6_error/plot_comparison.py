
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot comparisons of FP16 and FP32
"""

import numpy as np
import matplotlib as mpl
import scipy.ndimage as image
import h5py as h5
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_comparison(name, merr, dec, aspect, xtext, ytext, clip=0.005):
    
    """
    _____________________________Read Files____________________________________
    """

    file = h5.File('./' + name + '_csts.mat', 'r')
    dt = file['dt'][:] * dec
    NT = file['NT'][:] // dec
    NX = file['N'][-1]
    NZ = file['N'][0]
    ND = file['ND'][:]
    dh = file['dh'][:]
    nab = int(file['nab'][0][0])
    rec_pos = np.transpose(file['rec_pos'][:, ::dec])
    offmin = np.min(rec_pos[0, :]) / 1000
    offmax = np.max(rec_pos[0, :]) / 1000
    doff = rec_pos[0, 1] - rec_pos[0, 0]
    file.close()

    file = h5.File('./' + name + '_model.mat')
    vp = np.transpose(file['vp'])
    if ND == 3:
        vp = vp[nab:,:,vp.shape[-1]//2]
    file.close()

    cases = ['./' + name + 'fp32_dout.mat',
             './' + name + 'fp16_2_dout.mat',
             './' + name + 'fp16_3_dout.mat',
             './' + name + 'fp32hp_dout.mat',
             './' + name + 'fp32holberg_dout.mat']

    datas = []
    for case in cases:
        file = h5.File(case, 'r')
        datas.append(np.transpose(file['pout'][:, ::dec]))
        file.close()

    """
    _____________________________Plot______________________________________
    """

    fig = plt.figure(figsize=(18 / 2.54, 22 / 2.54))
    gridspec.GridSpec(10, 6)

    ax = plt.subplot2grid((10, 6), (0, 0), colspan=6, rowspan=2)

    im = ax.imshow(vp / 1000, interpolation='bilinear',
                   cmap=plt.get_cmap('jet'),
                   extent=[0, (NX + 1) * dh / 1000, (NZ + 1) * dh / 1000, 0],
                   aspect = aspect)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('z (km)')
    ax.set_title('a)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clr = plt.colorbar(im, cax=cax)
    cax.xaxis.tick_top()
    cax.set_xlabel('km/s', labelpad=8)
    cax.xaxis.set_label_position('top')

    vmin = np.min(datas[0]) * clip
    vmax = np.max(datas[0]) * clip
    vmin = -vmax

    labels = ['b)', 'c)' , 'd)', 'e)', 'f)', 'g)']
    toplot = [datas[0],
              (datas[0] - datas[1]) * merr,
              (datas[0] - datas[2]) * merr,
              (datas[0] - datas[4]) * merr,
              (datas[3] - datas[1]) * merr,
              (datas[3] - datas[2]) * merr]

    for ii, data in enumerate(toplot):
        
        ax = plt.subplot2grid((10, 6), (2 + (ii // 3) * 4, 2 * ii % 6),
                              colspan=2, rowspan=4)
        im = ax.imshow(data,
                       interpolation='none',
                       vmin=vmin, vmax=vmax,
                       cmap=plt.get_cmap('RdGy'),
                       extent=[offmin, offmax, NT * dt, 0],
                       aspect='auto')
        plt.tight_layout()
        ax.set_xlabel('offset (km)')
        ax.set_ylabel('Time (s)')
        ax.set_title(labels[ii])
        if ii != 0:
            ax.text(xtext, ytext, "Error X %d" % merr,
                    ha='right', weight='bold')

    plt.savefig('compare_acc_' + name + '.png', dpi=600)
    plt.savefig('compare_acc_' + name + '_lowres.png', dpi=100)

    print(np.sum((datas[0] - datas[1]) ** 2) / np.sum(datas[0] ** 2))
    print(np.sum((datas[0] - datas[2]) ** 2) / np.sum(datas[0] ** 2))
