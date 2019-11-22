#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Comparison with the analytical solution
"""
import numpy as np
from SeisCL import SeisCL
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
import os
from shutil import copyfile
import h5py as h5

def analytic_visco_pointz(vp, vs, rho, dt, rec_pos, wavelet):
    """
    Analytic solution for a point force in the z direction in an infinite
    homogeneous space

     vp: P-wave velocity
     vs: S-wave velocity
     rho: density
     dt: time step size
     rec_pos: a list of [ [x,y,z] ] for each receiver position
     wavelet: the wavelet signal

     The analytic solution can be found in:
     Gosselin-Cliche, B., & Giroux, B. (2014).
     3D frequency-domain finite-difference viscoelastic-wave modeling
     using weighted average 27-point operators with optimal coefficients.
     Geophysics, 79(3), T169-T188. doi: 10.1190/geo2013-0368.1
    """

    nt = wavelet.shape[0]
    nrec = len(rec_pos)
    F = np.fft.fft(wavelet)
    omega = 2*np.pi*np.fft.fftfreq(F.shape[0], dt)

    Vx = np.zeros([nt, nrec], dtype=np.complex128)
    Vy = np.zeros([nt, nrec], dtype=np.complex128)
    Vz = np.zeros([nt, nrec], dtype=np.complex128)

    y = 0
    for ii in range(1, nt):

        nu = vp ** 2 * rho
        mu = vs ** 2 * rho

        kp = omega[ii] / np.sqrt(nu / rho)
        ks = omega[ii] / np.sqrt(mu / rho)

        for jj in range(nrec):
            x, y, z = rec_pos[jj]

            R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            amp = F[ii] / (4.0 * np.pi * rho * R ** 5 * omega[ii] ** 2)

            Vx[ii, jj] = amp * x * z * (
                                        (R ** 2 * kp ** 2
                                         - 3.0 - 3.0 * 1j * R * kp)
                                        * np.exp(-1j * kp * R)

                                        + (3.0 + 3.0 * 1j * R * ks
                                           - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        )


            Vy[ii, jj] = amp * y * z * (
                                        (R ** 2 * kp ** 2
                                         - 3 - 3 * 1j * R * kp)
                                        * np.exp(-1j * kp * R) +

                                        (3 + 3 * 1j * R * ks
                                         - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        )

            Vz[ii, jj] = amp * (
                                (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                 * (np.exp(-1j * kp * R) - np.exp(-1j * ks * R))

                                + (z ** 2 * R ** 2 * kp ** 2
                                    + 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * kp) * np.exp(-1j * kp * R)

                                + ((x ** 2 + y ** 2) * R ** 2 * ks ** 2
                                    - 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * ks) * np.exp(-1j * ks * R)
                                )

    vx = np.real(nt * np.fft.ifft(Vx, axis=0))
    vy = np.real(nt * np.fft.ifft(Vy, axis=0))
    vz = np.real(nt * np.fft.ifft(Vz, axis=0))

    return vx, vy, vz

def fd_solution(seis, vp=3500, vs=2000, rho=2000,
                fp16=0, dir="z"):

    workdir = "./seiscl"
    fileout = "./" + seis.file + "_fp%d_dir%s.mat" % (fp16, dir)
    if not os.path.isdir(workdir):
        os.mkdir(workdir)
    nbuf = seis.csts['FDORDER']*2
    nab = seis.csts['nab']
    dh = seis.csts['dh']
    N = seis.csts['N']
    seis.csts['FP16'] = fp16
    if dir == "z":
        sz = (nab + nbuf) * dh
        sy = N[1] // 2 * dh
        sx = N[2] // 2 * dh
        offmin = 5 * dh
        offmax = (N[0] - nab - nbuf) * dh - sz
        gz = np.arange(sz + offmin, sz + offmax, dh)
        gx = gz * 0 + sx
        gy = gz * 0 + sy
    else:
        sz = N[0] // 2 * dh
        sy = N[1] // 2 * dh
        sx = (nab + nbuf) * dh
        offmin = 5 * dh
        offmax = (N[2] - nab - nbuf) * dh - sx
        gx = np.arange(sx + offmin, sx + offmax, dh)
        gz = gx * 0 + sz
        gy = gx * 0 + sy

    seis.src_pos_all = np.stack([[sx], [sy], [sz], [0], [2]], axis=0)

    gsid = gz * 0
    gid = np.arange(0, len(gz))
    seis.rec_pos_all = np.stack([gx, gy, gz, gsid, gid, gx * 0 + 2, gx * 0, gx * 0],
                            axis=0)

    if not os.path.isfile(fileout):
        vp_a = np.zeros([N[0], N[1], N[2]]) + vp
        vs_a = np.zeros([N[0], N[1], N[2]]) + vs
        rho_a = np.zeros([N[0], N[1], N[2]]) + rho

        seis.set_forward(seis.src_pos_all[3, :],
                         {"vp": vp_a, "rho": rho_a, "vs": vs_a},
                         workdir, withgrad=False)
        seis.execute(workdir)
        copyfile(workdir + "/" + seis.file_dout,
                 "./" + seis.file + "_fp%d_dir%s.mat" % (fp16, dir))
        data = seis.read_data(workdir)
    else:
        mat = h5.File(fileout, 'r')
        data = []
        for word in seis.to_load_names:
            if word+"out" in mat:
                datah5 = mat[word+"out"]
                data.append(np.transpose(datah5))

    return data


def main(vp=3500, vs=2000, rho=2000,
         dt=0.6e-03, NT=3000, dh=6, f0=20, FDORDER=12,
         N = [700, 700], nab= 112): #700, 2400 nab 112 dh 6.25
    
    ntoplot = 1
    fp16s= [1, 2, 3]
    seis = SeisCL()
    seis.csts['no_use_GPUs'] = np.array([0])
    seis.csts['dt'] = dt
    seis.csts['NT'] = NT
    seis.csts['dh'] = dh
    seis.csts['f0'] = f0
    seis.csts['freesurf'] = 0
    seis.csts['FDORDER'] = FDORDER
    seis.csts['MAXRELERROR'] = 1
    seis.csts['MAX'] = FDORDER
    seis.csts['abs_type'] = 2
    seis.csts['seisout'] = 1
    seis.csts['ND'] = 3
    seis.csts['nab'] = nab
    seis.csts['abpc'] = 3 # 3 nab 112
    workdir = "./seiscl"
    if not os.path.isdir(workdir):
        os.mkdir(workdir)
    
    fig, axs = plt.subplots(ntoplot * 3, 2, figsize=(16 / 2.54, 10 / 2.54))
    t = np.arange(0, NT-1) * dt
    freqs = np.fft.fftfreq(NT-1, dt)
    titles = [["a)", "b)", "c)"], ["d)", "e)", "f)"], ["g)", "h)", "j)"]]
    labels = ["Analytic", "Numeric", "Error"]
    plots = [[] for _ in range(3)]
    for fp16 in fp16s:
    # Recordings in z
        seis.csts['N'] = np.array([N[0], N[1], N[1]])
        data = fd_solution(seis, vp=vp, vs=vs, rho=rho,
                            fp16=fp16, dir="z")[2][:NT-1, :]
        seis.csts['NT'] = 2*NT
        src = seis.ricker_wavelet()
        seis.csts['NT'] = NT
        gx = seis.rec_pos_all[0, [0,-1]]
        gy = seis.rec_pos_all[1, [0,-1]]
        gz = seis.rec_pos_all[2, [0,-1]]
        sx = seis.src_pos_all[0, :]
        sy = seis.src_pos_all[1, :]
        sz = seis.src_pos_all[2, :]

        rec_pos = [[x, y, z] for x, y, z in zip(gx - sx, gy - sy, gz - sz)]
        vx, vy, vz = analytic_visco_pointz(vp, vs, rho, dt, rec_pos, src)



        vz_true = vz[1:NT, :]-vz[:NT-1, :]
        vz_true = vz_true / np.max(vz_true)
        vz_num = data / np.max(data)
        if fp16 == 1:
            vref = vz_num
            err = np.sum((vz_true[:,-1] - vref[:,-1])**2) / np.sum((vz_true[:,-1])**2) * 100
            print("Error due to dispersion for fp32 = %f %%" % err)
        else:
            err = np.sum((vz_num[:,-1] - vref[:,-1])**2) / np.sum((vref[:,-1])**2) * 100
            print("Error between fp16 = %d and fp32 = %f %%" % (fp16, err))

        axs[fp16-1, 0].plot(t, (vz_true[:,-1]-vz_num[:, -1])*10, "g")
        axs[fp16-1, 0].plot(t, vz_true[:,-1], 'k')
        axs[fp16-1, 0].plot(t, vz_num[:, -1], "r")
        axs[fp16-1, 0].set_xlabel("Time (s)")
        axs[fp16-1, 0].set_ylabel("Amplitude")
        axs[fp16-1, 0].set_title(titles[0][fp16-1])
        axs[fp16-1, 0].text(1.85, 0.0027, "Error X 10", ha='right', weight='bold')
        
        Vz_true = np.fft.fft(vz_true[:, -1], axis=0)
        Vz_num = np.fft.fft(vz_num[:, -1], axis=0)
        plots[2], = axs[fp16-1, 1].loglog(freqs[:NT//2], 2*np.abs(Vz_true-Vz_num)[:NT//2], "g")
        plots[0], = axs[fp16-1, 1].loglog(freqs[:NT//2], 2*np.abs(Vz_true)[:NT//2], "k")
        plots[1], = axs[fp16-1, 1].loglog(freqs[:NT//2], 2*np.abs(Vz_num)[:NT//2], "r")
        axs[fp16-1, 1].set_ylim([10**-10, 10**0])
        axs[fp16-1, 1].set_xlabel("Frequency (Hz)")
        axs[fp16-1, 1].set_ylabel("Amplitude")
        axs[fp16-1, 1].set_title(titles[1][fp16-1])
        #axs[fp16-1, 1].set_xscale('log', basex=2)
        axs[fp16-1, 1].set_yscale('log', basey=2)
        axs[fp16-1, 1].set_yticks([2**-2, 2**-10, 2**-18, 2**-26, 2**-34]   )

        if fp16==1:
            axs[fp16-1, 1].legend(plots[0:3], labels[0:3], loc='upper right',bbox_to_anchor=(1.05, 1.5), framealpha=1)


    plt.tight_layout()
    plt.savefig('analytic.eps')
    plt.savefig('analytic_lowres.eps')

if __name__ == "__main__":
    main()
