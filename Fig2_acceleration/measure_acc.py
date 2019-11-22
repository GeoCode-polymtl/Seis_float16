#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to measure acceleration of FP16 in 2D
Arguments: Arg1: name of the device
           Arg2: FP16 compute capacity of the device(0: no support, 1: support)
           Arg3: Number of dimension (2: 2D, 3: 3D)
"""
import os
import numpy as np
import sys
import h5py as h5
from SeisCL import SeisCL
import argparse

"""
    _________________________________Std in_____________________________________
"""
# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("name",
                    type=str,
                    help="Name of the device"
                    )
parser.add_argument("FP16compute",
                    type=int,
                    help="FP16 compute capacity (0: no support, 1:  support)"
                    )
parser.add_argument("ND",
                    type=int,
                    help="Number of dimension (2: 2D, 3: 3D)"
                    )
args = parser.parse_args()
appendname = args.name
FP16compute = args.FP16compute
ND = args.ND

filepath = os.getcwd() + '/'
filename = filepath + "times_" + str(ND) +"D" + appendname + ".mat"

"""
_____________________Simulation constants input file___________________________
"""
seis = SeisCL()
seis.csts['ND'] = ND  # Flag for dimension. 3: 3D, 2: 2D P-SV,  21: 2D SH
seis.csts['dh'] = 10  # Grid spatial spacing
seis.csts['dt'] = 0.0008  # Time step size
seis.csts['NT'] = 1000  # Number of time steps
seis.csts['freesurf'] = 0  # Include a free surface at z=0: 0: no, 1: yes
seis.csts['FDORDER'] = 4  # Order of the finite difference stencil.
seis.csts['f0'] = 7.5  # Central frequency of the source
seis.csts['abs_type'] = 0  # Absorbing boundary type (none here)
seis.csts['seisout'] = 0  # Output seismograms (none here)


"""
_________________________Sources and receivers_________________________________
"""
# Dummy receiver positions
gx = np.arange(0, 2) * seis.csts['dh']
gz = np.arange(0, 2) * seis.csts['dh']

# We model 4 shots with dummy positions
for ii in range(0, 4, 1):
    idsrc = seis.src_pos_all.shape[1]
    toappend = np.zeros((5, 1))
    toappend[3, :] = idsrc
    seis.src_pos_all = np.append(seis.src_pos_all, toappend, axis=1)


    toappend = np.stack([gx,
                         gx * 0,
                         gz,
                         gz * 0 + idsrc,
                         np.arange(0, len(gx)) + seis.rec_pos_all.shape[1] + 1,
                         gx * 0 + 2,
                         gx * 0,
                         gx * 0], 0)
    seis.rec_pos_all = np.append(seis.rec_pos_all, toappend, axis=1)


"""
    ___________________________Comparison__________________________________
"""
# The model sizes that are tested
if ND == 2:
    sizes = np.arange(512, 8192, 64)
elif ND == 3:
    sizes = np.arange(64, 600, 8)

# Cases tested: FP16=1 -> FP32 storage, FP32 compute
#               FP16=2 -> FP16 storage, FP32 compute
#               FP16=3 -> FP16 storage, FP16 compute
cases = {'timesFP1': 1, 'timesFP2': 2}
if FP16compute==1:
    cases['timesFP3'] = 3

# read checkpoint files if some results already exist
if os.path.isfile(filename):
    times = h5.File(filename, 'r+')
    imin = 9999
    for case in cases:
        imin = np.min([np.argmax(times[case][:] == 0), imin])
else:
    times = h5.File(filename, 'w')
    imin = 0
    for case in cases:
        times[case] = np.zeros(len(sizes))

# Perform computations, for all FP16 cases and all model sizes
for ii in range(imin, len(sizes)):
    size = sizes[ii]
    thissize = tuple([size]*ND)
    model = {}
    model['vp'] = np.zeros(thissize) + 3500
    model['rho'] = np.zeros(thissize) + 2000
    model['vs'] = np.zeros(thissize) + 2000

    for case in cases:
        seis.csts['FP16'] = cases[case]
        seis.csts['N'] = np.array(thissize)
        seis.set_forward(seis.src_pos_all[:,3], model, withgrad=False)
        stdout = seis.execute()
        
        # Extract time stepping run time from SeisCL output
        runtime = float(stdout.splitlines()[-5].split(': ')[1])
        times[case][ii] = runtime
        times.flush()
        print("size %d, FP16 %d : %f s" % (size, cases[case], times[case][ii]))
times.close()

