# adding pyMCR folder to the system path
import sys
sys.path.insert(0, 'pyMCR')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg

#####################################################################################################################
#####################################################################################################################
#
# Walk-through:
# 1. fit_model() uses least_squares(P, _calc_residuals) where P is a fitting parameter vector, and _calc_residuals() is the cost function.
# 2. _calc_residuals(P) calls either calc_model_deltaA(P) to calculate the model deltaA matrix from the normalized species decays in 'global fitting'
#    or calculates the residuals in the fit of the normalized model species decays to the normalized MCR-ALS populations.
# 3. calc_model_deltaA(P) calls calc_species_decays(P) to construct the species decays using the model by solving the differential equations and
#    convolving with IRF in _convolve_with_IRF()
#
#####################################################################################################################
#####################################################################################################################


class FitModel:
    
    def __init__(self, TA, model, irf=0.3, tzero=0.0):
        
        self.TA = TA
        self.model = model
        self.lower_bounds = np.concatenate(([0.01, -1.0],self.model.lower_bounds_K))
        self.upper_bounds = np.concatenate(([50.0, 1.0],self.model.upper_bounds_K))
        self.irf = irf
        self.tzero = tzero
        self.optimized_result = []
        self.fit_errors = np.ones(len(self.model.K)+1)*np.nan
        self.mean_squared_error = np.nan
        self.covariance_matrix = np.ones((len(self.fit_errors), len(self.fit_errors)))*np.nan
        self.fittype = 'global' # fit type can be: {'global', 'MCR-ALS'}
        
        # initialize mcrar object for multivariate curve resolution fitting
        self.mcrals = McrAR(c_regr='OLS',
                      st_regr='OLS',
                      c_constraints=[ConstraintNonneg()], # constraint on the populations to be non-negative
                      st_constraints=[],
                      max_iter=100)
        
    def _convolve_with_IRF(self, kinetic_traces, irf, tzero):
        
        ######################################################################
        # INPUT: kinetic traces are arranged in a (D,W)
        # matrix where W is the number of kinetic traces selected and D is the
        # length of the delay vector
        ######################################################################
        
        # find time-zero index in delay vector
        t0_index = np.argmin(abs(self.TA.delay-tzero))
        
        # zero the kinetic traces prior to time zero (effectively applying a step function)
        kinetic_traces[0:t0_index, :] = 0.0
        
        # create linearly sampled time axis for convolution with Gaussian
        dt = (self.TA.delay[1]-self.TA.delay[0])/3 # for interpolation, use 1/3 of a step in the delay vector
        delay_positive = np.arange(tzero, self.TA.delay[-1]+dt, dt)
        delay_negative = np.arange(-self.TA.delay[-1], tzero, dt)
        delay = np.concatenate((delay_negative, delay_positive))
        
        # create Gaussian IRF
        b = 4*np.log(2.0)/(irf**2)
        gaussian_irf = np.exp(-b * delay**2)
        gaussian_irf = gaussian_irf / np.sum(gaussian_irf) # normalize the Gaussian
        
        # create array that will hold the convolved kinetic traces
        kinetic_traces_convolved = np.zeros((self.TA.delay.size, kinetic_traces.shape[1]))
        
        # loop over selected number of kinetic traces
        for i in range(kinetic_traces.shape[1]):
            
            # interpolate each kinetic trace to the linearly sampled delay
            kinetic_trace_interp1d_func = interp1d(self.TA.delay, kinetic_traces[:,i], bounds_error=False, fill_value=0.0)
            kinetic_trace_interp1d = kinetic_trace_interp1d_func(delay_positive)
            kinetic_trace_interp1d = np.concatenate((np.zeros(delay_negative.size), kinetic_trace_interp1d))
            
            # convolve with IRF
            kinetic_trace_interp1d_convolved = convolve(kinetic_trace_interp1d, gaussian_irf, mode='same')
            
            # interpolate result back to log-sampled delay
            kinetic_trace_convolved_interp1d_func = interp1d(delay, kinetic_trace_interp1d_convolved, bounds_error=False, fill_value=0.0)
            kinetic_traces_convolved[:,i] = kinetic_trace_convolved_interp1d_func(self.TA.delay)
        
        return kinetic_traces_convolved
    
    def _calc_residuals(self, P):
        
        # calculate residuals
        if self.fittype == 'global':
            
            # calculate the model deltaA matrix using the fitting parameter vector P
            model_deltaA = self.calc_model_deltaA(P)
            residuals_as_matrix = self.TA.deltaA - model_deltaA
            
        elif self.fittype == 'MCR-ALS':
            
            mcrals_populations = self.mcrals.C_opt_
            model_populations = self.calc_species_decays(P)
            
            # normalize MCR-ALS and model populations
            for i in range(mcrals_populations.shape[1]):
                mcrals_populations[:,i] = mcrals_populations[:,i]/np.max(np.abs(mcrals_populations[:,i]))

            residuals_as_matrix = mcrals_populations - model_populations
        
        # flatten residuals to a 1d vector
        residuals_as_vector = np.reshape(residuals_as_matrix, newshape=residuals_as_matrix.size)
        
        # remove NaNs
        residuals_as_vector = residuals_as_vector[~np.isnan(residuals_as_vector)]
        
        return residuals_as_vector
    
    def _calculate_fit_error(self):
        
        # get the Jacobian matrix from the best fit
        jacobian = self.optimized_result.jac
        
        # set zero values to a small epsilon so that the covariance is not singular
        jacobian[jacobian==0.0] = 1e-10
        
        # calculate mean-squared error
        self.mean_squared_error = np.mean(self.optimized_result.fun**2)
        print('Mean-squared error = ' + str(self.mean_squared_error))
        
        # from the Jacobian and MSE calculate the covariance matrix
        self.covariance_matrix = np.linalg.inv(jacobian.T.dot(jacobian)) * self.mean_squared_error
        
        # get the parameter error estimations from the covariance matrix
        return np.sqrt(np.diagonal(self.covariance_matrix))
    
    def calc_model_deltaA(self, P):
        
        # calculate the model species (population) decays
        species_decays = self.calc_species_decays(P)
        
        # calculate coefficient matrix (C) of converting from deltaA (A) to species (S): A=C*S -> C=A*inv(S)
        pseudo_inv_species_decays = np.linalg.pinv(species_decays)
        coeffs = np.matmul(pseudo_inv_species_decays, self.TA.deltaA.T)
        
        # calculate model deltaA matrix using this coefficient matrix.
        model_deltaA = np.matmul(species_decays, coeffs).T
        
        return model_deltaA
        
    def calc_species_decays(self, P):
        
        # extract fitting parameters from fitting parameter vector P
        irf = P[0]
        tzero = P[1]
        K = P[2:]
        
        # get time span vector: (first_delay, last_delay) to use when solving the model diff. eq.
        t_span = (self.TA.delay[0], self.TA.delay[-1])
        
        # calculate the species decay traces (as column vectors).
        # The solve_ivp method 'LSODA' appears to be much faster (~0.01 sec) than the default 'RK45' (~0.2 sec)
        if self.model.type == 'diffeq':
            species_decays = solve_ivp(lambda t,y: self.model.diffeq(t,y,K), t_span, self.model.initial_populations, t_eval=self.TA.delay, method='LSODA').y.transpose()
        elif self.model.type == 'other':
            species_decays = self.model.get_species_decay(self.TA.delay, K)
            
        # colvolve with IRF
        species_decays = self._convolve_with_IRF(species_decays, irf, tzero)
        
        # normalize species
        for i in range(len(species_decays[0,:])):
            species_decays[:,i] = species_decays[:,i]/np.max(np.abs(species_decays[:,i]))
        
        return species_decays
    
    def get_experimental_kinetic_traces(self, wavelengths):
        
        # make sure at least one wavelength is provided by the user
        if wavelengths==[]:
            return
        
        # get wavelength indices
        wavelength_inds = [np.argmin(abs(wavelength-self.TA.wavelength)) for wavelength in np.array(wavelengths)]

        # get the column vectors of the data decay traces at the selected wavelengths
        experiment_kinetic_traces = self.TA.deltaA[wavelength_inds,:].transpose()
        
        return experiment_kinetic_traces
    
    def get_model_kinetic_traces(self, P, wavelengths):
        
        # make sure at least one wavelength is provided by the user
        if wavelengths==[]:
            return
        
        # get wavelength indices
        wavelength_inds = [np.argmin(abs(wavelength-self.TA.wavelength)) for wavelength in np.array(wavelengths)]

        # get the column vectors of the model decay traces at the selected wavelengths
        model_kinetic_traces = self.calc_model_deltaA[wavelength_inds,:].transpose()
        
        return model_kinetic_traces
    
    def calc_model_species_spectra(self, P):
        
        # calculate the model species (population) decays
        species_decays = self.calc_species_decays(P)
        
        # calculate the (pseudo-) inverse of the species decay matrix (S) that is needed to retreive the species' spectra (Epsilon)
        # deltaA(lambda,delay) = Epsilon(lambda,#species) * transpose(S(delay,#species))
        pseudo_inv_transpose_species_decays = np.linalg.pinv(species_decays.transpose())
        
        # calculate the model species spectra
        model_species_spectra = np.matmul(self.TA.deltaA, pseudo_inv_transpose_species_decays)
        
        return model_species_spectra
    
    def normalize_species_spectra(self, model_species_spectra):
        
        for i in range(model_species_spectra.shape[1]):
            model_species_spectra[:,i] = model_species_spectra[:,i]/np.max(np.abs(model_species_spectra[:,i]))
            
        return model_species_spectra
    
    def calc_model_deltaA_residual_matrix(self, P):
        
        # calculate deltaA residual matrix
        deltaA_residuals = self.TA.deltaA - self.calc_model_deltaA(P)
        
        return deltaA_residuals
        
    def fit_model(self):
        
        # build array of initial guesses of the fitting parameters
        P0 = np.concatenate(([self.irf, self.tzero], self.model.K))
        
        # set bounds on fitting parameters and make sure that for each parameter, lower and upper bounds are different
        lower_bounds = list(map(lambda x: x-1e-10, self.lower_bounds))
        upper_bounds = list(map(lambda x: x+1e-10, self.upper_bounds))
        fit_param_bounds = (lower_bounds, upper_bounds)
        
        # use least-squares to calculate fitting parameters for the vector of fitting parameters P=[irf,tzero,K]
        self.optimized_result = least_squares(self._calc_residuals,
                                              x0=P0,
                                              x_scale=np.concatenate(([self.irf, 0.1], self.model.K)),
                                              bounds=fit_param_bounds,
                                              loss='soft_l1',
                                              ftol = 1e-12, 
                                              xtol = 1e-12, 
                                              gtol = 1e-12, 
                                              max_nfev = 50*len(P0), # maximum number of function evaluations
                                              verbose=2 # 0=no printing of steps during computation, 2=printing of steps during computation
                                              )
        self.irf = self.optimized_result.x[0]
        self.tzero = self.optimized_result.x[1]
        self.model.K = self.optimized_result.x[2:]
        
        self.fit_errors = self._calculate_fit_error()
        
    def change_initial_guess(self, initial_guess):
        
        self.irf = initial_guess[0]
        self.tzero = initial_guess[1]
        self.model.K = initial_guess[2:]
        self.fit_errors = np.ones(len(self.model.K)+1)*np.nan
        self.mean_squared_error = np.nan
        self.covariance_matrix = np.ones((len(self.fit_errors), len(self.fit_errors)))*np.nan
        
    def change_model(self, new_model):
        
        # after changing a model, the fitmodel object is almost completely "reset"
        # with its __init__ method.
        # Some model-unrelated attributes remain after changing the model:
        # TA, irf, tzero, mcrals
        
        mcrals = self.mcrals
        self.__init__(self.TA, new_model, self.irf, self.tzero)
        self.mcrals = mcrals
#####################################################################################################################
#####################################################################################################################        
        