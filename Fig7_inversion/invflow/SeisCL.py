# -*- coding: utf-8 -*-
"""
Interface to SeisCL for Tensorflow
"""
import hdf5storage as h5mat
import h5py as h5
import numpy as np
import math
from invflow.Forward import Forward, FclassError


class SeisCL(Forward):
    """ A forward class that implements a Tensorflow operator for SeisCL
        (https://github.com/gfabieno/SeisCL.git)
    """
    def __init__(self):   

        self.file='model' #General filename/path models and parameters (see setter)
        self.file_datalist=None #The data file containing all of the dataset (see setter)
        self.progname='SeisCL_MPI'
        self.input_residuals=False

        #_____________________Simulation constants _______________________
        self.csts={}
        self.csts['N']=np.array([200,150])              #Grid size in X
        self.csts['ND']=2                #Flag for dimension. 3: 3D, 2: 2D P-SV,  21: 2D SH
        self.csts['dh']=10                #Grid spatial spacing
        self.csts['dt']=0.0008           # Time step size
        self.csts['NT']=875              #Number of time steps
        self.csts['freesurf']=0          #Include a free surface at z=0: 0: no, 1: yes
        self.csts['FDORDER']=8           #Order of the finite difference stencil. Values: 2,4,6,8,10,12
        self.csts['MAXRELERROR']=1       #Set to 1
        self.csts['L']=0                 #Number of attenuation mechanism (L=0 elastic)
        self.csts['f0']=15               #Central frequency for which the relaxation mechanism are corrected to the righ velocity
        self.csts['FL']=np.array(15)     #Array of frequencies in Hz of the attenuation mechanism
        
        self.csts['src_pos']=np.empty((5,0)) #Position of each shots. 5xnumber of sources. [sx sy sz srcid src_type]. srcid is the source number (two src with same srcid are fired simulatneously) src_type: 1: Explosive, 2: Force in X, 3: Force in Y, 4:Force in Z
        self.csts['rec_pos']=np.empty((8,0)) #Position of the receivers. 8xnumber of traces. [gx gy gz srcid recid Not_used Not_used Not_used]. srcid is the source number recid is the trace number in the record
        self.csts['src']=np.empty((self.csts['NT'],0))            #Source signals. NTxnumber of sources
        
        self.csts['abs_type']=1          #Absorbing boundary type: 1: CPML, 2: Absorbing layer of Cerjan
        self.csts['VPPML']=3500          #Vp velocity near CPML boundary
        self.csts['NPOWER']=2            #Exponent used in CMPL frame update, the larger the more damping
        self.csts['FPML']=15              #Dominant frequency of the wavefield
        self.csts['K_MAX_CPML']=2        #Coeffienc involved in CPML (may influence simulation stability)
        self.csts['nab']=32              #Width in grid points of the absorbing layer
        self.csts['abpc']=6              #Exponential decay of the absorbing layer of Cerjan et. al.
        self.csts['pref_device_type']=4  #Type of processor used: 2: CPU, 4: GPU, 8: Accelerator
        self.csts['nmax_dev']=9999       #Maximum number of devices that can be used
        self.csts['no_use_GPUs']=np.empty( (1,0) )  #Array of device numbers that should not be used for computation
        self.csts['MPI_NPROC_SHOT']=1    #Maximum number of MPI process (nodes) per shot involved in domain decomposition
        
        self.csts['back_prop_type']=2    #Type of gradient calculation: 1: backpropagation (elastic only) 2: Discrete Fourier transform
        self.csts['param_type']=0        #Type of parametrization: 0:(rho,vp,vs,taup,taus), 1:(rho, M, mu, taup, taus), 2:(rho, Ip, Is, taup, taus)
        self.csts['gradfreqs']=np.empty((1,0)) #Array of frequencies in Hz to calculate the gradient with DFT
        self.csts['tmax']=0         #Maximum time for which the gradient is to be computed
        self.csts['tmin']=0              #Minimum time for which the gradient is to be computed
        self.csts['scalerms']=0          #Scale each modeled and recorded traces according to its rms value, then scale residual by recorded trace rms
        self.csts['scalermsnorm']=0      #Scale each modeled and recorded traces according to its rms value, normalized
        self.csts['scaleshot']=0         #Scale all of the traces in each shot by the shot total rms value
        self.csts['fmin']=0              #Maximum frequency for the gradient computation
        self.csts['fmax']=0              #Minimum frequency for the gradient computation
        self.csts['mute']=None           #Muting matrix 5xnumber of traces. [t1 t2 t3 t4 flag] t1 to t4 are mute time with cosine tapers, flag 0: keep data in window, 1: mute data in window
        self.csts['weight']=None         # NTxnumber of geophones or 1x number of geophones. Weight each sample, or trace, according to the value of weight for gradient calculation.
        
        self.csts['gradout']=0           #Output gradient 1:yes, 0: no
        self.csts['Hout']=0              #Output approximate Hessian 1:yes, 0: no
        self.csts['gradsrcout']=0        #Output source gradient 1:yes, 0: no
        self.csts['seisout']=1           #Output seismograms 1:yes, 0: no
        self.csts['resout']=0            #Output residuals 1:yes, 0: no
        self.csts['rmsout']=0            #Output rms value 1:yes, 0: no
        self.csts['movout']=0            #Output movie 1:yes, 0: no
        self.csts['restype']=0           #Type of costfunction 0: raw seismic trace cost function. No other available at the moment
        self.csts['inputres']=0          #Input the residuals for gradient computation

         
        # Variable for data maipulation, always equal to the self.csts member
        # (see setter)
        self.src_pos=np.empty((5,0))
        self.rec_pos=np.empty((8,0))
        self.src=np.empty((self.csts['NT'],0))
        
        #For data manipulation inside this class
        self.src_pos_all=np.empty((5,0))
        self.rec_pos_all=np.empty((8,0))
        self.allshotids=[]
        self.srcids=np.empty((0,0))
        self.recids=np.empty((0,0))
        self.read_src=False

        self.mute=None
        self.mute_window=np.empty((4,0))
        self.mute_picks=np.empty((1,0))
        self.offmin=-float('Inf')
        self.offmax=float('Inf')

    #_____________________Setters _______________________
    
    
    #When setting a file for the datalist, load the datalist from it  
    @property
    def file_datalist(self):
        return self.__file_datalist

    @file_datalist.setter
    def file_datalist(self, file_datalist):
        self.__file_datalist=file_datalist
        if self.file_datalist:
            mat = h5.File(file_datalist,'r')
            fields={'src_pos':'src_pos_all','rec_pos':'rec_pos_all'}
            for word in fields.keys():
                data=mat[word]
                setattr(self, fields[word], np.transpose(data[:,:]) )
            self.allshotids = self.src_pos_all[3,:]
    
    #Params returns the list of parameters required by the simulation constants
    @property
    def params(self):
        if self.csts['param_type']==0:
            params=['vp','vs','rho']
        elif self.csts['param_type']==1:
            params=['M','mu','rho']
        elif self.csts['param_type']==2:
            params=['Ip','Is','rho']  
        else:
            raise NotImplementedError()
        if self.csts['L']>0:
            params.append('taup')
            params.append('taus')
        
        return params


    #Given the general filename, set the specific filenames of each files
    @property
    def file(self):
        return self.__file

    @file.setter
    def file(self, file):
        self.__file=file
        self.file_model=file+"_model.mat"
        self.file_csts=file+"_csts.mat"
        self.file_dout=file+"_dout.mat"
        self.file_gout=file+"_gout.mat"
        self.file_rms=file+"_rms.mat"
        self.file_movout=file+"_movie.mat"
        self.file_din=file+"_din.mat"
        self.file_res = file + "_res.mat"

    #The variable src_pos, rec_pos and src must always be reflected 
    #in self.csts to be written
    @property
    def src_pos(self):
        return self.csts['src_pos']

    @src_pos.setter
    def src_pos(self, src_pos):
        if not type(src_pos) is np.ndarray:
            raise FclassError('src_pos must be a numpy arrays')
        if src_pos.shape[0]!=5:
            raise FclassError('src_pos must be a numpy arrays with dim 5x num of src')
        self.csts['src_pos']=src_pos
        
    @property
    def rec_pos(self):
        return self.csts['rec_pos']

    @rec_pos.setter
    def rec_pos(self, rec_pos):
        if not type(rec_pos) is np.ndarray:
            raise FclassError('rec_pos must be a numpy arrays')
        if rec_pos.shape[0]!=8:
            raise FclassError('rec_pos must be a numpy arrays with dim 8x num of rec')
        self.csts['rec_pos']=rec_pos
        
    @property
    def src(self):
        return self.csts['src']

    @src.setter
    def src(self, src):
        if not type(src) is np.ndarray:
            raise FclassError('src must be a numpy arrays')
        if src.shape[0]!=self.csts['NT']:
            raise FclassError('src must be a numpy arrays with dim NT x num of src')
        self.csts['src']=src

    @property
    def to_load_names(self):
        toload = []
        if self.csts['seisout'] == 1:
            if self.csts['ND'] == 2:
                toload = ["vx", "vz"]
            if self.csts['ND'] == 21:
                toload = ["vy"]
            if self.csts['ND'] == 3:
                toload = ["vx", "vy", "vz"]
        if self.csts['seisout'] == 2:
            toload = ["p"]

        return toload

            
    def set_forward(self, jobids, params, workdir, withgrad=True):

        self.srcids = np.empty((0, 0), 'int')
        self.recids = np.empty((0, 0), 'int')
        if len(np.atleast_1d(jobids)) > 1:
            shots = np.sort(jobids)
        else:
            shots = np.atleast_1d(jobids)
        for shot in shots:
            self.srcids = np.append(self.srcids,
                                    np.where(self.src_pos_all[3,:].astype(int) == shot) )
            self.recids = np.append(self.recids,
                                    np.where(self.rec_pos_all[3,:].astype(int) == shot) )
        if len(self.srcids) <= 0:
            raise FclassError('No shot found')
        self.src_pos = self.src_pos_all[:,self.srcids]
        self.rec_pos = self.rec_pos_all[:,self.recids]
        self.prepare_data()
        if withgrad:
            self.csts['gradout'] = 1
        else:
            self.csts['gradout'] = 0
        self.write_csts(workdir)
        self.write_model(workdir, params)

    def set_backward(self, workdir, residuals):
        data = {}
        for n, word in enumerate(self.to_load_names):
           data[word+"res"] = residuals[n]
        h5mat.savemat(workdir+self.file_res,
                      data,
                      appendmat=False,
                      format='7.3',
                      store_python_metadata=True,
                      truncate_existing=True)
        self.csts['gradout'] = 1
        self.write_csts(workdir)

    def callcmd(self,workdir, osname='Darwin'):
        cmd= self.progname+' '+workdir+'/'+self.file+' '+self.file_din
        return cmd
    
    def read_data(self, workdir):

        try:
            mat = h5.File(workdir+self.file_dout, 'r')
        except :
            raise FclassError('Could not read data')
        output = []
        for word in self.to_load_names:
            if word+"out" in mat:
                datah5 = mat[word+"out"]
                data = np.transpose(datah5)
                output.append(data)  
                
        if not output:
            raise FclassError('Could not read data: variables not found')
            
        return output
    
    def read_grad(self, workdir, param_names):
        toread = ['grad'+name for name in param_names]
        try:
            mat = h5.File(workdir+self.file_gout, 'r')
            output = [np.transpose(mat[v]) for v in toread]
#            mat = h5mat.loadmat(workdir+self.file_gout, variable_names=toread)
        except :
            raise FclassError('Could not read grad')
        
        
        return output 
    
    def read_Hessian(self,  workdir, param_names):
        toread = ['H' + name for name in param_names]
        try:
            mat = h5.File(workdir+self.file_gout, 'r')
            output = [np.transpose(mat[v]) for v in toread]
        except :
            raise FclassError('Could not read Hessian')

        return output

    def read_rms(self, workdir):
        try:
            mat = h5mat.loadmat(workdir+self.file_rms, variable_names=['rms','rms_norm'])
        except:
            raise FclassError('Forward modeling failed, courld not read rms\n')
            
        return mat['rms']/mat['rms_norm'], mat['rms_norm']
            
    def write_data(self, workdir, data):
        if 'src_pos' not in data:
            data['src_pos']=self.src_pos
        if 'rec_pos' not in data:
            data['rec_pos']=self.rec_pos
        if 'src' not in data:
            data['src']=self.src
        h5mat.savemat(self.file_din,
                      data,
                      appendmat=False,
                      format='7.3',
                      store_python_metadata=True,
                      truncate_existing=True)
            

            
    def read_csts(self, workdir):
       try:
           mat = h5mat.loadmat(workdir+self.file_csts, variable_names=[param for param in self.csts])
           for word in mat:
               if word in self.csts:
                   self.csts[word]= mat[word]
       except (h5mat.lowlevel.CantReadError,NotImplementedError):
           raise FclassError('could not read parameter file \n')
                
    def read_srcs(self):
        try:
            mat = h5.File(self.file_datalist,'r')
            data=mat['src']
            setattr(self, 'src', np.transpose(data[self.srcids,:]) )
        except :
            raise FclassError('could not read src\n')                       
                
    def write_csts(self, workdir):
        h5mat.savemat(workdir+self.file_csts, self.csts , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
    
#    def read_model(self, workdir):
#        output=[[]]*len(self.params)
#        for (ii,param) in enumerate(self.params):
#            try:
#                mat = h5mat.loadmat(workdir+self.file_model, variable_names=[param])
#                output[ii]=mat[param]
#            except (h5mat.lowlevel.CantReadError,NotImplementedError):
#                raise FclassError('could not read param '+param+'\n') 
#        return output
        
    def write_model(self, workdir, params):
        for param in self.params:
            if param not in params:
                raise FclassError('Parameter with name %s required by SeisCL\n'%param)
        h5mat.savemat(workdir+self.file_model, params , appendmat=False, format='7.3', store_python_metadata=True, truncate_existing=True)
        
#    def update_model(self, workdir, params):
#        h5mat.savemat(workdir+self.file_model, params , appendmat=True, format='7.3', store_python_metadata=True, truncate_existing=False)
                            
    


    def prepare_data(self):
        
        validrec=[]
        for ii in range(0, self.rec_pos.shape[1] ):
            srcid=np.where(self.src_pos[3,:]==self.rec_pos[3,ii]) 
            offset=np.sqrt(np.square(self.rec_pos[0,ii]-self.src_pos[0,srcid])+np.square(self.rec_pos[1,ii]-self.src_pos[1,srcid])+np.square(self.rec_pos[2,ii]-self.src_pos[2,srcid]))
            if offset<=self.offmax and offset>=self.offmin:
                validrec.append(ii)
        self.rec_pos=self.rec_pos[:,validrec]
        self.recids=self.recids[validrec]
        self.rec_pos[4,:]=[x+1 for x in self.recids]
        
        
        if self.read_src:
            self.read_srcs()
        else:
            self.ricker()
      
        if np.any(self.mute_window):
           self.mute=np.transpose(np.tile(self.mute_window, (self.rec_pos.shape[1],1 )) );
           if np.any(self.mute_picks):
               for ii in range(0, self.recids.size):
                   self.mute[:3,ii]=self.mute[:3,ii]+self.mute_picks[self.recids[ii]]

    def ricker(self):
        if self.src.shape[1] == 0:
            self.src=np.empty((self.csts['NT'],0))
        ns0=self.src.shape[1]
        nstot=self.src_pos.shape[1]
        if ns0>nstot:
            self.src=np.empty((self.csts['NT'],0))
            ns0=self.src.shape[1]
        
        tmin=-1.5/self.csts['f0']
        t = np.linspace(tmin, self.csts['NT']*self.csts['dt']+tmin, num=self.csts['NT']) #was -tmin, wrong!

        ricker = ((1.0 - 2.0*(np.pi**2)*(self.csts['f0']**2)*(t**2)) 
                * np.exp(-(np.pi**2)*(self.csts['f0']**2)*(t**2))  )

        self.src=np.stack( [ricker]*nstot  , 1)
            
    def add_src_pos(self,sx,sy,sz,srcnum,srctype,*args):
        
        if isinstance(sx, list):
           toappend=np.zeros((5,len(sx))) 
        else:
           toappend=np.zeros((5,1)) 
        
        toappend[0,:]=sx
        toappend[1,:]=sy
        toappend[2,:]=sz
        toappend[3,:]=srcnum
        toappend[4,:]=srctype

        self.src_pos=np.append(self.src_pos, toappend  , axis=1)
        self.src_pos_all=self.src_pos
        
        if args:
            self.src=np.append(self.src, args[0]  , axis=1)
        else:
            self.ricker()
                 
    def add_rec_pos(self,gx,gy,gz,srcnum):
        
        if isinstance(gx, list):
           ng= len(gx)
           toappend=np.zeros((8,ng)) 
        else:
           toappend=np.zeros((8,1)) 
           ng=1
        
        toappend[0,:]=gx
        toappend[1,:]=gy
        toappend[2,:]=gz
        toappend[3,:]=srcnum
        toappend[4,:]=[self.rec_pos.shape[1]+ii+1 for ii in range(0,ng)]

        self.rec_pos=np.append(self.rec_pos, toappend  , axis=1)     
        self.rec_pos_all=self.rec_pos
        

        
