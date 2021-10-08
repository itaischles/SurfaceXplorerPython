import numpy as np
import scipy.interpolate
import os.path

class TransientAbsorption:
    
    def __init__(self, filepath):
        
        self.change_file(filepath)
        self.rawdata = self.load_raw_data(filepath)
        self.rawdata[np.isnan(self.rawdata)] = 0.0
        self.delay = np.array(self.rawdata[0,1:], copy=True)
        self.wavelength = np.array(self.rawdata[1:,0], copy=True)
        self.deltaA = np.array(self.rawdata[1:,1:], copy=True)
        self.chirp_coeffs = [0.,0.,0.]
        self.pret0_background = np.zeros(self.wavelength.shape)
    
    def load_raw_data(self, filepath):
        
        # find if there are footer rows in the file such as in Femto-A or Femto-B data
        with open(filepath) as f:
            max_rows_to_read = 0
            s = f.readline()
            while True:
                if (s=='') or (s[0]=='f'):
                        break
                max_rows_to_read = max_rows_to_read+1
                s = f.readline()
        
        rawdata = np.genfromtxt(filepath, delimiter='\t', max_rows=max_rows_to_read)
        if rawdata.ndim != 2: # This happens if the delimiter is not '\t' (tab) but ',' (comma)
            rawdata = np.genfromtxt(filepath, delimiter=',', max_rows=max_rows_to_read)
            
        return rawdata
    
    def crop(self, delay_range, wavelength_range, method):
        
        wavelength_inds = np.arange(self.wavelength.size)
        wavelength_mask = (self.wavelength >= wavelength_range[0]) & (self.wavelength <= wavelength_range[1])
        wavelength_inds = wavelength_inds[wavelength_mask]
        
        delay_inds = np.arange(self.delay.size)
        delay_mask = (self.delay >= delay_range[0]) & (self.delay <= delay_range[1])
        delay_inds = delay_inds[delay_mask]
        
        # Method='keep' keeps the selected region.
        # Method='delete' deletes the selected region by replacing it with zeros. 
        if method=='keep':
            self.deltaA = self.deltaA[wavelength_mask,:][:,delay_mask]
            self.wavelength = self.wavelength[wavelength_mask]
            self.delay = self.delay[delay_mask]
        elif method=='delete':
            self.deltaA[wavelength_inds[0]:wavelength_inds[-1], delay_inds[0]:delay_inds[-1]] = np.nan
            
    def reset_TA_data(self):
        
        self.delay = np.array(self.rawdata[0,1:], copy=True)
        self.wavelength = np.array(self.rawdata[1:,0], copy=True)
        self.deltaA = np.array(self.rawdata[1:,1:], copy=True)
        
    def change_file(self, new_filepath):
        
        # example: if new_filepath = 'C:/Users/Experiments/data.csv' then:
        # self.filepath = 'C:/Users/Experiments'
        # self.basename = 'data'
        
        self.filepath = os.path.split(new_filepath)[0] # the path where the TA file (.csv) is saved. only folders
        self.basename = os.path.splitext(os.path.basename(new_filepath))[0] # the basename of the file (no folders, no .csv)
        
    def chirp_correct(self):
        
        for wavelength_index in np.arange(self.wavelength.size):
            chirp = self.chirp_fit_func(self.wavelength[wavelength_index], *self.chirp_coeffs)
            interp_func = scipy.interpolate.interp1d(self.delay, self.deltaA[wavelength_index,:], bounds_error=False, fill_value='extrapolate')
            self.deltaA[wavelength_index,:] = interp_func(self.delay+chirp)
        
    def chirp_fit_func(self, x, a, b, c):
        return a*x**2 + b*x + c
    
    def subtract_pret0_background(self):
        
        self.deltaA = self.deltaA - np.tile(self.pret0_background, (self.delay.size,1)).transpose()
