import tkinter as tk

import numpy as np
import scipy.ndimage
import scipy.interpolate

import os
import time

# import sub-GUIs
from plot_area_gui import PlotAreaFrame
from fitting_gui import FittingUserInputFrame, EditInitialGuessGui
from misc_guis import FilePathFrame, LogFrame
from data_correction_guis import ChirpCorrectGui, SVDGui, SubtractBackgroundGui

# import TA and fitting classes
from transient_absorption_class import TransientAbsorption
from global_fitting import FitModel

# import model library
import model_library

#####################################################################################################################
#####################################################################################################################

class SurfaceXplorerPythonGui(tk.Tk):
    
    def __init__(self):
        
        super().__init__() # parent class (=tk.Toplevel) initialization
        
        self.title('SurfaceXplorerPython')
        self.option_add('*tearOff', False) # ALWAYS use this line if putting in menubar!!!!!
        
        # load the models from the model library file (should be located in this folder!)
        self.fit_model_classes, self.fit_model_names = self.read_model_library()
        
        # initialize boolean variables
        self.fitting_mode = tk.BooleanVar()
        self.fitting_mode.set(False)
        
        # Create frames that sit inside this main frame
        self.filepath_frame = FilePathFrame(self)
        self.plot_area_frame = PlotAreaFrame(self)
        self.log_frame = LogFrame(self)
        self.fitting_user_input_frame = FittingUserInputFrame(self, self.fit_model_names)

        # arrange frames in a grid
        self.filepath_frame.grid(row=0,column=0,columnspan=2,sticky='nsew')
        self.fitting_user_input_frame.grid(row=1,column=0,sticky='nsew')
        self.plot_area_frame.grid(row=1,column=1,sticky='nsew')
        self.log_frame.grid(row=2,column=0,columnspan=2,sticky='nsew')
        
        # create Menubar
        self.create_menubar()
        
        # bindings
        self.fitting_user_input_frame.fit_button.configure(command=self.fit_to_model)
        
        # write Welcome message in log
        self.log_frame.update_log('Welcome to SurfaceXplorerPython!')
        
    def create_menubar(self):
        
        # create the root (abstract) menubar to serve as parent
        self.menubar = tk.Menu(self)
        
        # create abstract menu children (not the hierarchy!)
        self.file_menu = tk.Menu(self.menubar)
        self.surface_menu = tk.Menu(self.menubar)
        self.crop_menu = tk.Menu(self.surface_menu)
        self.filter_menu = tk.Menu(self.surface_menu)
        self.fitting_menu = tk.Menu(self.menubar)
        self.compare_menu = tk.Menu(self.menubar)
        
        # create menu items with labels and separators. Order matters!!!
        self.menubar.add_cascade(menu=self.file_menu, label='File')
        self.menubar.add_cascade(menu=self.surface_menu, label='Surface')
        self.menubar.add_cascade(menu=self.fitting_menu, label='Fitting')
        self.menubar.add_cascade(menu=self.compare_menu, label='Compare')
        self.file_menu.add_command(label='Open...', command=self.open_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Save As...', command=self.save_as)
        self.file_menu.add_command(label='Export data', command=self.export_data)
        self.surface_menu.add_cascade(menu=self.crop_menu, label='Crop')
        self.crop_menu.add_command(label='Keep selected', command=self.crop_keep_selected)
        self.crop_menu.add_command(label='Delete selected', command=self.crop_delete_selected)
        self.surface_menu.add_command(label='Subtract background...', command=self.subtract_background)
        self.surface_menu.add_command(label='Chirp correct...', command=self.chirp_correct)
        self.surface_menu.add_command(label='SVD filter', command=self.SVD_filter)
        self.surface_menu.add_cascade(menu=self.filter_menu, label='Smooth')
        self.filter_menu.add_command(label='Apply Gaussian filter', command=self.apply_gaussian_filter)
        self.filter_menu.add_command(label='Apply Median filter', command=self.apply_median_filter)
        self.filter_menu.add_command(label='Apply DCT filter', command=self.apply_dct_filter)
        self.surface_menu.add_command(label='Quick create POIs (fs)', command=self.quick_create_POIs_fs)
        self.surface_menu.add_command(label='Quick create POIs (ns)', command=self.quick_create_POIs_ns)
        self.surface_menu.add_separator()
        self.surface_menu.add_command(label='Clear all POIs', command=self.clear_all_POIs)       
        self.surface_menu.add_command(label='Undo all changes', command=self.undo_all_changes)
        self.fitting_menu.add_checkbutton(label='Fitting mode', onvalue = True, offvalue = False, variable=self.fitting_mode, command=self.refresh_plots)
        self.fitting_menu.add_command(label='Edit initial guess', command=self.edit_initial_guess)
        self.compare_menu.add_command(label='Compare kinetics...', command=self.compare_kinetics)
        self.compare_menu.add_command(label='Compare spectra...', command=self.compare_spectra)
        self.compare_menu.add_command(label='Compare surfaces...', command=self.compare_surfaces)
        
        # disable some menus at startup
        self.file_menu.entryconfig('Save As...', state='disabled')
        self.file_menu.entryconfig('Export data', state='disabled')
        self.menubar.entryconfig('Surface', state='disabled')
        self.menubar.entryconfig('Fitting', state='disabled')
        self.menubar.entryconfig('Compare', state='disabled')
        
        # show the menubar at the top of the main GUI
        self.config(menu=self.menubar)
    
    def open_file(self):
        
        # open the file and initialize TA structure
        filepath = tk.filedialog.askopenfilename(
            master = self,
            title = 'Select file to open...',
            multiple = False,
            filetypes = [('CSV files','*.csv')]
            )
        if filepath=='':
            return
        self.TA = TransientAbsorption(filepath)
        
        # clear POI table
        self.plot_area_frame.POI_table.clear_table()
        
        # enable menu items
        self.file_menu.entryconfig('Save As...', state='normal')
        self.file_menu.entryconfig('Export data', state='normal')
        self.menubar.entryconfig('Surface', state='normal')
        self.menubar.entryconfig('Fitting', state='normal')
        self.menubar.entryconfig('Compare', state='normal')
        
        # enable adding POIs
        self.event_add_POI = self.plot_area_frame.fig.canvas.mpl_connect('button_press_event', self.add_POI_button_click)
        
        # disable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='disabled')
        
        # enable prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # create a fit model object
        fit_model_class = getattr(model_library, self.fit_model_classes[0])
        self.fitmodel = FitModel(self.TA, fit_model_class())
        
        # re-initialize fitting GUI
        self.fitting_user_input_frame.selected_model.set(self.fit_model_names[0])
        self.fitting_user_input_frame.fit_params_table.populate(self.fitmodel)
        
        # Refresh plots
        self.refresh_plots()
        
        # update path presented to user
        self.filepath_frame.set_new_filepath(filepath)
        
        # update log
        self.log_frame.update_log('Opened file: ' + filepath)
        
        # bindings
        self.fitting_user_input_frame.selected_model.trace_add('write', self.change_model)
        
    def save_as(self):
        
        # get the file path from the user
        filepath = tk.filedialog.asksaveasfilename(
            master = self,
            title = 'Select file name for saving...',
            defaultextension = [('CSV files','*.csv')],
            filetypes = [('CSV files','*.csv')]
            )
        if filepath=='':
            return
        
        # prepare data for saving
        deltaA = self.TA.deltaA
        delay = self.TA.delay
        wavelength = self.TA.wavelength
        temp1 = np.concatenate(([0],delay),axis=0)[None,:]
        temp2 = np.concatenate((wavelength[:,None],deltaA),axis=1)
        TA_to_save = np.concatenate((temp1,temp2), axis=0)
        
        # save the file
        with open(filepath, 'wb') as f:
            np.savetxt(f, TA_to_save, delimiter=',', fmt='%.6e')
            
        # set the filepath property of the TA object to the new file path
        self.TA.change_file(filepath)
        
        # update path presented to user
        self.filepath_frame.set_new_filepath(filepath)
        
        # update log
        self.log_frame.update_log('Saved file sucessfully to: ' + filepath)
    
    def refresh_plots(self):
        
        # update the fitting model object
        wavelengths = self.plot_area_frame.POI_table.get_all_wavelengths()
        self.fitmodel.get_experimental_kinetic_traces(wavelengths)
        
        # Refresh plots
        if self.fitting_mode.get()==True:
            self.plot_area_frame.refresh_fig1(self.TA, self.fitmodel)
            self.plot_area_frame.refresh_fig2(self.TA, self.fitmodel)
            self.plot_area_frame.refresh_fig3(self.TA, self.fitmodel)
            self.plot_area_frame.refresh_fig4(self.TA, self.fitmodel)
        else:
            self.plot_area_frame.refresh_fig1(self.TA, [])
            self.plot_area_frame.refresh_fig2(self.TA, [])
            self.plot_area_frame.refresh_fig3(self.TA, [])
            self.plot_area_frame.refresh_fig4(self.TA, [])
        
        # update the plot toolbar
        self.plot_area_frame.toolbar.update()
    
    def chirp_correct(self):
        
        # open chirp-correct GUI for user and wait for user to close that window or accept the chirp.
        # Get the chirp coefficients upon closing
        self.TA.chirp_coeffs = ChirpCorrectGui(self, self.TA).get_chirp_coeffs()
        
        # if all chirp coefficients are 0 it means that the user cancelled chirp correction
        if all([x==0 for x in self.TA.chirp_coeffs]):
            self.log_frame.update_log('Data was NOT chirp corrected. Cancelled by user')
        else:
            # chirp correct the TA data
            self.TA.chirp_correct()
            
            # clear all POIs if they were present before chirp correction
            self.plot_area_frame.POI_table.clear_table()
            
            # update log
            self.log_frame.update_log('TA data chirp corrected')
            
            # disable fitting button
            self.fitting_user_input_frame.fit_button.configure(state='disabled')
            
            # enable prompting user to add POIs to enable fitting
            self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # refresh plots
        self.refresh_plots()
        
    def SVD_filter(self):
        
        # open SVD GUI for user and wait for user to close that window.
        SVD_components, SVD_filtered_deltaA = SVDGui(self, self.TA).get_SVD_filtered_deltaA()
        if len(SVD_components) == 0:
            # update log
            self.log_frame.update_log('Data was NOT SVD filtered. Cancelled by user')
        else:
            self.TA.deltaA = SVD_filtered_deltaA
            # update log
            self.log_frame.update_log('TA data SVD filtered using '+str(len(SVD_components))+' component(s)')
            # clear all POIs if they were present before SVD filtering
            self.plot_area_frame.POI_table.clear_table()
            # disable fitting button
            self.fitting_user_input_frame.fit_button.configure(state='disabled')
            # enable prompting user to add POIs to enable fitting
            self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
            # refresh plots
            self.refresh_plots()
        
    def crop_keep_selected(self):
        
        # clear POI table
        self.plot_area_frame.POI_table.clear_table()
        
        # reset colors
        self.plot_area_frame.POI_table.setup_colors(num_colors=15)
        
        # get cropping ranges
        delay_range = self.plot_area_frame.ax1.get_ylim()
        wavelength_range = self.plot_area_frame.ax1.get_xlim()
                    
        # apply the crop function to the TA surface
        self.TA.crop(delay_range, wavelength_range, method='keep')
        
        # disable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='disabled')
        
        # enable prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # Refresh plots, reset cursor object, and bind it to mouse movement over axis
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('TA data cropped. New delay range={}, New wavelength range={}.'.format(delay_range,wavelength_range))
    
    def crop_delete_selected(self):
        
        # clear POI table
        self.plot_area_frame.POI_table.clear_table()
        
        # reset colors
        self.plot_area_frame.POI_table.setup_colors(num_colors=15)
        
        # get cropping ranges
        delay_range = self.plot_area_frame.ax1.get_ylim()
        wavelength_range = self.plot_area_frame.ax1.get_xlim()
        
        # apply the crop function to the TA surface
        self.TA.crop(delay_range, wavelength_range, method='delete')
        
        # disable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='disabled')
        
        # enable prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # Refresh plots, reset cursor object, and bind it to mouse movement over axis
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('TA data cropped. Deleted delay range={}, Deleted wavelength range={}.'.format(delay_range,wavelength_range))
            
    def add_POI_button_click(self, event):
        
        # if the toolbar is in either 'zoom' or 'pan' mode, do not add POI
        if (self.plot_area_frame.toolbar.mode != 'zoom rect') and (self.plot_area_frame.toolbar.mode != 'pan/zoom'):
        
            # add point to POI table
            wavelength, delay = event.xdata, event.ydata
            self.plot_area_frame.POI_table.insert_poi(wavelength, delay)
            
            # enable fit button
            self.fitting_user_input_frame.fit_button.configure(state='normal')
            
            # hide prompting user to add POIs to enable fitting
            self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid_forget()
            
            # refresh plots
            self.refresh_plots()
    
    def clear_all_POIs(self):
        
        # clear POI table
        self.plot_area_frame.POI_table.clear_table()
        
        # reset colors
        self.plot_area_frame.POI_table.setup_colors(num_colors=15)
        
        # disable fit button
        self.fitting_user_input_frame.fit_button.configure(state='disabled')
        
        # enable prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # Refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('Cleared all POIs')
        
    def undo_all_changes(self):
        
        if tk.messagebox.askyesno("Undo all changes", "This will undo all changes to the TA data. Continue?"):
        
            # clear POI table
            self.plot_area_frame.POI_table.clear_table()
            
            # reset TA
            self.TA.reset_TA_data()
            
            # disable fitting button
            self.fitting_user_input_frame.fit_button.configure(state='disabled')
            
            # enable prompting user to add POIs to enable fitting
            self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
            # Refresh plots
            self.refresh_plots()
            
            # update log
            self.log_frame.update_log('Undo all changes')
            
    def apply_gaussian_filter(self):
        
        # apply Gaussian filter
        self.TA.deltaA = scipy.ndimage.gaussian_filter(self.TA.deltaA, sigma=0.75)
        
        # Refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('TA data smoothed using Gaussian filter')
        
    def apply_median_filter(self):
        
        # apply median filter
        self.TA.deltaA = scipy.ndimage.median_filter(self.TA.deltaA, 3)
        
        # Refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('TA data smoothed using median filter')
        
    def apply_dct_filter(self):
        
        # calculate DCT
        deltaA_DCT = scipy.fft.dct(scipy.fft.dct(self.TA.deltaA, axis=0), axis=1)
        
        # filter high-frequency spectral components
        deltaA_DCT[40:,:] = 0.0
        
        # calculate inverse DCT
        self.TA.deltaA = scipy.fft.idct(scipy.fft.idct(deltaA_DCT, axis=0), axis=1)
        
        # Refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('TA data smoothed using DCT filter')
    
    def quick_create_POIs_fs(self):
        
        # clear old POIs
        self.clear_all_POIs()

        # add new POIs
        requested_delays = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
        spectral_range = self.plot_area_frame.ax1.get_xlim()
        wavelengths = np.linspace(spectral_range[0], spectral_range[1], requested_delays.size)
        for i in range(requested_delays.size):
            wavelength = wavelengths[i]
            delay = self.TA.delay[np.argmin(abs(requested_delays[i]-self.TA.delay))]
            self.plot_area_frame.POI_table.insert_poi(wavelength, delay)
        
        # enable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='normal')
        
        # hide prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid_forget()
            
        # refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('Added new automatic POIs')
        
    def quick_create_POIs_ns(self):
        
        # clear old POIs
        self.clear_all_POIs()

        # add new POIs
        requested_delays = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
        spectral_range = self.plot_area_frame.ax1.get_xlim()
        wavelengths = np.linspace(spectral_range[0], spectral_range[1], requested_delays.size)
        for i in range(requested_delays.size):
            wavelength = wavelengths[i]
            delay = self.TA.delay[np.argmin(abs(requested_delays[i]-self.TA.delay))]
            self.plot_area_frame.POI_table.insert_poi(wavelength, delay)
        
        # enable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='normal')
        
        # hide prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid_forget()
            
        # refresh plots
        self.refresh_plots()
        
        # update log
        self.log_frame.update_log('Added new automatic POIs')
        
    def export_data(self):
        
        # if no POIs are present do not export anything
        if self.plot_area_frame.POI_table.num_POIs == 0:
            
            # update log
            self.log_frame.update_log('Export failed: no POIs found')
            return
        
        # extract data indices from POI table
        wavelength_indices = np.zeros(0)
        delay_indices = np.zeros(0)
        wavelengths = self.plot_area_frame.POI_table.get_all_wavelengths()
        delays = self.plot_area_frame.POI_table.get_all_delays()
        for (wavelength,delay) in zip(wavelengths, delays):
            wavelength_indices = np.append(wavelength_indices, np.argmin(abs(wavelength-self.TA.wavelength)))
            delay_indices = np.append(delay_indices, np.argmin(abs(delay-self.TA.delay)))
        wavelength_indices = [int(x) for x in wavelength_indices][::-1]
        delay_indices = [int(x) for x in delay_indices][::-1]
        
        # initialize delay and wavelength vectors as first columns
        first_col_kinetics = np.expand_dims(np.concatenate((np.zeros(1),self.TA.delay)),axis=1)
        first_col_spectra = np.expand_dims(np.concatenate((np.zeros(1),self.TA.wavelength)),axis=1)
        
        ########################################
        # Prepare arctan scaling of TA surface #
        ########################################
        
        deltaA_scaled = np.arctan(self.TA.deltaA/np.max(self.TA.deltaA))
        deltaA_scaled = np.concatenate((np.asmatrix(self.TA.delay),deltaA_scaled))
        deltaA_scaled = np.concatenate((np.asmatrix(np.insert(self.TA.wavelength, 0, 0)).T,deltaA_scaled),axis=1)
        
        ################
        # Prepare fits #
        ################
        
        fit_params = np.concatenate(([self.fitmodel.irf, self.fitmodel.tzero], self.fitmodel.model.K))
        deltaA_residuals = self.fitmodel.calc_model_deltaA_residual_matrix(fit_params)
        model_kinetics = self.fitmodel.calc_model_kinetic_traces(fit_params)
        model_spectra = self.fitmodel.calc_model_species_spectra(fit_params)
        model_populations = self.fitmodel.calc_species_decays(fit_params)
        
        # add the labels for each kinetic trace (label=wavelength)
        model_kinetics = np.concatenate((np.array(self.fitmodel.check_wavelengths,ndmin=2),model_kinetics), axis=0)
        
        # add delay and wavelength vectors as first columns
        model_spectra = np.concatenate((first_col_spectra[1:], model_spectra), axis=1)
        model_kinetics = np.concatenate((first_col_kinetics, model_kinetics), axis=1)
        model_populations = np.concatenate((first_col_kinetics[1:], model_populations), axis=1)
        
        ################
        # Prepare POIs #
        ################
        
        # arrange data for saving
        kinetics = self.TA.deltaA[wavelength_indices, :].transpose()
        spectra = self.TA.deltaA[:, delay_indices]
        
        # add the labels for each spectrum (label=delay) and kinetic trace (label=wavelength)
        kinetics = np.concatenate((np.array(self.TA.wavelength[wavelength_indices],ndmin=2),kinetics), axis=0)
        spectra = np.concatenate((np.array(self.TA.delay[delay_indices],ndmin=2),spectra), axis=0)
        
        # add delay and wavelength vectors as first columns
        kinetics = np.concatenate((first_col_kinetics, kinetics), axis=1)
        spectra = np.concatenate((first_col_spectra, spectra), axis=1)
        
        ############################
        # Prepare folders and save #
        ############################
        
        # create folders that contain the exported data if they haven't been created yet
        if not os.path.isdir(self.TA.filepath + '/Exports/POIs'):
            os.makedirs(self.TA.filepath + '/Exports/POIs')
        if not os.path.isdir(self.TA.filepath + '/Exports/Fits'):
            os.makedirs(self.TA.filepath + '/Exports/Fits')
        if not os.path.isdir(self.TA.filepath + '/Exports/Scaled TA surface'):
            os.makedirs(self.TA.filepath + '/Exports/Scaled TA surface')

        # save the POI kinetics
        kinetics_filepath = self.TA.filepath + '/Exports/POIs/' + self.TA.basename +'__POI_kinetics.csv'
        with open(kinetics_filepath, 'wb') as f:
            np.savetxt(f, kinetics, delimiter=',', fmt='%.6e')
        
        # save the POI spectra
        spectra_filepath = self.TA.filepath + '/Exports/POIs/' + self.TA.basename +'__POI_spectra.csv'
        with open(spectra_filepath, 'wb') as f:
            np.savetxt(f, spectra, delimiter=',', fmt='%.6e')
            
        # save the fitted species spectra
        species_spectra_filepath = self.TA.filepath + '/Exports/Fits/' + self.TA.basename +'__normalized_species_spectra.csv'
        with open(species_spectra_filepath, 'wb') as f:
            np.savetxt(f, model_spectra, delimiter=',', fmt='%.6e')
            
        # save the fitted kinetics
        model_kinetics_filepath = self.TA.filepath + '/Exports/Fits/' + self.TA.basename +'__fitted_kinetics.csv'
        with open(model_kinetics_filepath, 'wb') as f:
            np.savetxt(f, model_kinetics, delimiter=',', fmt='%.6e')
            
        # save the populations
        model_populations_filepath = self.TA.filepath + '/Exports/Fits/' + self.TA.basename +'__populations.csv'
        with open(model_populations_filepath, 'wb') as f:
            np.savetxt(f, model_populations, delimiter=',', fmt='%.6e')
            
        # save the deltaA residuals
        deltaA_residuals_filepath = self.TA.filepath + '/Exports/Fits/' + self.TA.basename +'__deltaA_residuals.csv'
        with open(deltaA_residuals_filepath, 'wb') as f:
            np.savetxt(f, deltaA_residuals, delimiter=',', fmt='%.6e')
            
        # save the deltaA matrix in scaled-arctan form
        deltaA_scaled_filepath = self.TA.filepath + '/Exports/Scaled TA surface/' + self.TA.basename +'__deltaA_arctan_scaled.csv'
        with open(deltaA_scaled_filepath, 'wb') as f:
            np.savetxt(f, deltaA_scaled, delimiter=',', fmt='%.6e')
            
        # save the fit report file
        fit_report_filepath = self.TA.filepath + '/Exports/Fits/' + self.TA.basename +'__fit_report.txt'
        with open(fit_report_filepath, 'w') as f:
            f.write('##################\n')
            f.write('### FIT REPORT ###\n')
            f.write('##################\n\n')
            f.write('Model: ' + self.fitmodel.model.name + '\n\n')
            f.write('Initial populations: ' + str(self.fitmodel.model.initial_populations) + '\n\n')
            f.write('IRF = ' + str(self.fitmodel.irf) + ' +- ' + str(self.fitmodel.fit_errors[0]) + '\n')
            f.write('t0 = ' + str(self.fitmodel.tzero) + ' +- ' + str(self.fitmodel.fit_errors[1]) + '\n')
            for name, value, error in zip(self.fitmodel.model.parameter_names, self.fitmodel.model.K, self.fitmodel.fit_errors[2:]):
                f.write(name + ': ' + str(value) + ' +- ' + str(error) + '\n')
            f.write('\nCovariance matrix: \n')
            f.write(str(self.fitmodel.covariance_matrix) + '\n\n')
            f.write('Mean-squared error: \n')
            f.write(str(self.fitmodel.mean_squared_error) + '\n')
            
        # update log
        self.log_frame.update_log('Exported data to "/Exports" folder')
    
    
    def subtract_background(self):
        
        # open pret0 background subtraction GUI for user and wait for user to close that window or accept.
        # Get the pre-t0 background upon closing
        self.TA.pret0_background = SubtractBackgroundGui(self, self.TA).get_pret0_background()
    
        # subtract the pre-t0 background
        self.TA.subtract_pret0_background()
            
        # update log
        self.log_frame.update_log('Pre-t0 background subtracted from TA data')
        
        # disable fitting button
        self.fitting_user_input_frame.fit_button.configure(state='disabled')
        
        # enable prompting user to add POIs to enable fitting
        self.fitting_user_input_frame.promt_user_to_add_POIs_label.grid(row=4,column=0,columnspan=2)
        
        # refresh plots
        self.refresh_plots()
        
    def read_model_library(self):
        
        with open('model_library.py','r') as f:
            
            fit_model_names = []
            fit_model_classes = []
            
            while True:
                line = f.readline()
                if not line:
                    break
                if line.replace(':','').replace('\n','').split(' ')[0]=='class':
                    fit_model_classes.append(line.replace(':','').replace('\n','').split(' ')[1])
                if line.replace(' ','').split('=')[0]=='self.name':
                    fit_model_names.append(line.replace(' ','').replace('\n','').replace("'",'').split('=')[1])
        
        print('Found the following models in the library: ')
        print(fit_model_names)
        print()
        
        return fit_model_classes, fit_model_names
     
    def change_model(self, *ignore):
        
        # get the new model from the combobox
        new_model_name = self.fitting_user_input_frame.selected_model.get()
        new_model_num_in_library = self.fit_model_names.index(new_model_name)
        
        # create a new fitmodel instance with the new model
        new_model_class = getattr(model_library, self.fit_model_classes[new_model_num_in_library])
        self.fitmodel = FitModel(self.TA, new_model_class())
        
        # refresh the user input frame
        self.fitting_user_input_frame.fit_params_table.populate(self.fitmodel)
        
        # refresh plots
        self.refresh_plots()
    
    def fit_to_model(self):
        
        # configure the fit button to show user fitting is in progress
        self.fitting_user_input_frame.fit_button.configure(text='Fitting, please wait...', state='disabled')
        
        # use update_idletasks() to actually change the button appearance,
        # otherwise the tkinter mainloop freezes while running the fitting function
        self.update_idletasks()
        
        # start the fitting
        start_time = time.time()
        self.fitmodel.fit_model()
        fitting_time = time.time()-start_time
        
        # Refresh plots
        self.refresh_plots()
        
        # refresh the parameters table
        self.fitting_user_input_frame.fit_params_table.populate(self.fitmodel)
        
        # configure the fit button to show user fitting is in progress
        self.fitting_user_input_frame.fit_button.configure(text='Fit', state='normal')
        
        # update log
        self.log_frame.update_log('Fitted the data to model '+self.fitting_user_input_frame.selected_model.get()+' in '+str(fitting_time)+' sec')
        
    def edit_initial_guess(self):
        
        # get new initial guess from the user
        initial_guess, lower_bounds, upper_bounds = EditInitialGuessGui(self, self.fitmodel).get_initial_guess()
        
        # update the fitmodel object with the new initial guess
        self.fitmodel.change_initial_guess(initial_guess)
        
        # update the fitmodel lower and upper bounds for fitting
        self.fitmodel.lower_bounds = lower_bounds
        self.fitmodel.upper_bounds = upper_bounds
        
        # refresh the parameters table
        self.fitting_user_input_frame.fit_params_table.populate(self.fitmodel)
        
        # Refresh plots
        self.refresh_plots()
        
        # refresh the parameters table
        self.fitting_user_input_frame.fit_params_table.populate(self.fitmodel)
        
        # update log
        self.log_frame.update_log('New initial guess of fitting parameters')
        
    def compare_kinetics(self):
        
        # open the file to be compared and initialize TA_compare structure
        filepath = tk.filedialog.askopenfilename(
            master = self,
            title = 'Select data to compare...',
            multiple = False,
            filetypes = [('CSV files','*.csv')]
            )
        if filepath=='':
            return
        self.TA_compare = TransientAbsorption(filepath)
        
        # # interpolate wavelength axis of self.TA_compare to that of self.TA
        # TA_interp2d_func = scipy.interpolate.interp2d(self.TA_compare.delay, self.TA_compare.wavelength, self.TA_compare.deltaA)
        # self.TA_compare.deltaA = np.flipud(TA_interp2d_func(self.TA.delay, self.TA.wavelength))

        # plot the comparison
        self.plot_area_frame.plot_compared_kinetics(self.TA, self.TA_compare)
        
    def compare_spectra(self):
        
        # open the file to be compared and initialize TA_compare structure
        filepath = tk.filedialog.askopenfilename(
            master = self,
            title = 'Select data to compare...',
            multiple = False,
            filetypes = [('CSV files','*.csv')]
            )
        if filepath=='':
            return
        self.TA_compare = TransientAbsorption(filepath)
        
        # # interpolate wavelength axis of self.TA_compare to that of self.TA
        # TA_interp2d_func = scipy.interpolate.interp2d(self.TA_compare.delay, self.TA_compare.wavelength, self.TA_compare.deltaA)
        # self.TA_compare.deltaA = np.flipud(TA_interp2d_func(self.TA.delay, self.TA.wavelength))

        # plot the comparison
        self.plot_area_frame.plot_compared_spectra(self.TA, self.TA_compare)
        
    def compare_surfaces(self):
        
        # open the file to be compared and initialize TA_compare structure
        filepath = tk.filedialog.askopenfilename(
            master = self,
            title = 'Select data to compare...',
            multiple = False,
            filetypes = [('CSV files','*.csv')]
            )
        if filepath=='':
            return
        self.TA_compare = TransientAbsorption(filepath)

        # plot the comparison
        self.plot_area_frame.plot_compared_surfaces(self.TA, self.TA_compare)
        
        
        
        
        