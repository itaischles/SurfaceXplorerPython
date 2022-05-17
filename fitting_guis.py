import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm # color maps
import numpy as np
import model_library

#####################################################################################################################
#####################################################################################################################

class FittingUserInputFrame(tk.Frame):

    def __init__(self, master, fit_model_classes, fitmodel=[]):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        self.fit_model_classes = fit_model_classes
        
        # initialize selection of number of species
        self.num_species_var = tk.StringVar(self, value='1')
        self.num_species_label = tk.Label(self, text='Num. species:')
        self.num_species_spinbox = tk.Spinbox(self, from_=1, to=5, increment=1, textvariable=self.num_species_var, state='disabled')
        
        # initialize model selection
        self.selected_model = tk.StringVar(self)
        self.model_combobox = ttk.Combobox(self, state='disabled')
        self.populate_model_combobox()
        self.model_label = ttk.Label(self, text='Model:')
        
        # initialize fitting parameter table
        self.fit_params_table = FitParamsTable(self, fitmodel)
        
        # initialize fit button
        self.fit_button = ttk.Button(self, text='Fit', state='disabled')
        
        # initialize radiobutton group that selects the type of fit ('global' or 'MCR-ALS')
        self.fittype_var = tk.IntVar()
        self.fittype_global_radiobutton = ttk.Radiobutton(self, text='Global fitting', variable=self.fittype_var, value=0, state='disabled')
        self.fittype_mcrals_radiobutton = ttk.Radiobutton(self, text='MCR-ALS fitting', variable=self.fittype_var, value=1, state='disabled')
        
        # use geometry manager to arrange widgets in user input frame
        self.num_species_label.grid(row=0,column=0,sticky='nw')
        self.num_species_spinbox.grid(row=0,column=1,sticky='ne')
        self.model_label.grid(row=1,column=0,sticky='nw')
        self.model_combobox.grid(row=1,column=1,sticky='ne')
        self.fit_params_table.grid(row=2,column=0,columnspan=2)
        self.fit_button.grid(row=3,column=0,columnspan=2)
        self.fittype_global_radiobutton.grid(row=4,column=0,columnspan=2)
        self.fittype_mcrals_radiobutton.grid(row=5,column=0,columnspan=2)
        
        # resizing
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=0)
        self.grid_rowconfigure(5, weight=0)
    
    def populate_model_combobox(self, *ignore):

        # initialize array that stores model names with correct number of species chosen by the user
        model_names = []
        
        try:
            # get the number of species
            num_species = int(self.num_species_var.get())
        except ValueError:
            return
        
        for i in range(len(self.fit_model_classes)):
            
            model = getattr(model_library, self.fit_model_classes[i])()
            
            # if the current fit model has the same number of species as that selected by the user, show it in the combobox
            model_num_species = len(model.initial_populations)
            if model_num_species == num_species:
                model_names.append(model.name)
                
        # change selected model to be the first on the new list
        self.selected_model.set(model_names[0])
        
        # populate the combobox
        self.model_combobox.configure(values=model_names,
                                      textvariable=self.selected_model,
                                      )
        
    def get_selected_fit_model(self):
        
        for i in range(len(self.fit_model_classes)):
            model = getattr(model_library, self.fit_model_classes[i])()
            if model.name == self.selected_model.get():
                return model
            
        # if no matching model found, return the first in the library
        return getattr(model_library, self.fit_model_classes[0])()
        
#####################################################################################################################
#####################################################################################################################

class FitParamsTable(ttk.Treeview):
    
    def __init__(self, master, fitmodel):
        
        # initialize treeview
        super().__init__(master,
                         columns=('col#1', 'col#2', 'col#3'),
                         show='headings',
                         height=7,
                         selectmode = 'browse'
                         )
        
        # set columns text and appearance
        self.heading('col#1', text='Paramter')
        self.heading('col#2', text='Value')
        self.heading('col#3', text='Error')
        self.column('col#1', width=80)
        self.column('col#2', width=80)
        self.column('col#3', width=80)
        
        self.populate(fitmodel)
        
    def populate(self, fitmodel):
        
        # clear the table
        for iid in self.get_children(): self.delete(iid)
        
        # go over all rows and insert names for fitting parameters K if fitting model supplied
        if fitmodel != []:
            for (name, value, error) in zip(fitmodel.model.parameter_names[::-1], fitmodel.model.K[::-1], fitmodel.fit_errors[::-1]):
                self.insert('', 0, values=(name, '{:.3f}'.format(value), '{:.3f}'.format(error)))
            # insert a row for time-zero
            self.insert('', 0, values=('t0', '{:.3f}'.format(fitmodel.tzero), '{:.3f}'.format(fitmodel.fit_errors[1])))
            # insert a row for IRF
            self.insert('', 0, values=('IRF', '{:.3f}'.format(fitmodel.irf), '{:.3f}'.format(fitmodel.fit_errors[0])))

#####################################################################################################################
#####################################################################################################################       

class EditInitialGuessGui(tk.Toplevel):
    
    def __init__(self, master, fitmodel):
        
        super().__init__(master)  # parent class (=tk.Toplevel) initialization
        
        self.label = []
        self.entry_values = []
        self.entry_lower_bounds = []
        self.entry_upper_bounds = []
        self.values = []
        self.lower_bounds = []
        self.upper_bounds = []
        
        # add headers
        tk.Label(self, text='Value', justify='center').grid(row=0,column=1)
        tk.Label(self, text='Lower bound', justify='center').grid(row=0,column=2)
        tk.Label(self, text='Upper bound', justify='center').grid(row=0,column=3)
        
        # add labels for IRF
        self.label.append(tk.Label(self, text='IRF', justify='left'))
        self.label[0].grid(row=1,column=0)
        
        # initialize string variables for the IRF
        self.values.append(tk.StringVar())
        self.values[0].set(str(fitmodel.irf))
        self.lower_bounds.append(tk.StringVar())
        self.lower_bounds[0].set(str(fitmodel.lower_bounds[0]))
        self.upper_bounds.append(tk.StringVar())
        self.upper_bounds[0].set(str(fitmodel.upper_bounds[0]))
        
        # add entry boxes for the IRF
        self.entry_values.append(tk.Entry(self, textvariable=self.values[0]))
        self.entry_values[0].grid(row=1,column=1)
        self.entry_lower_bounds.append(tk.Entry(self, textvariable=self.lower_bounds[0]))
        self.entry_lower_bounds[0].grid(row=1,column=2)
        self.entry_upper_bounds.append(tk.Entry(self, textvariable=self.upper_bounds[0]))
        self.entry_upper_bounds[0].grid(row=1,column=3)
        
        # add labels for time-zero
        self.label.append(tk.Label(self, text='t0', justify='left'))
        self.label[1].grid(row=2,column=0)
        
        # initialize string variables for the time-zero
        self.values.append(tk.StringVar())
        self.values[1].set(str(fitmodel.tzero))
        self.lower_bounds.append(tk.StringVar())
        self.lower_bounds[1].set(str(fitmodel.lower_bounds[1]))
        self.upper_bounds.append(tk.StringVar())
        self.upper_bounds[1].set(str(fitmodel.upper_bounds[1]))

        # add entry boxes for the time-zero
        self.entry_values.append(tk.Entry(self, textvariable=self.values[1]))
        self.entry_values[1].grid(row=2,column=1)
        self.entry_lower_bounds.append(tk.Entry(self, textvariable=self.lower_bounds[1]))
        self.entry_lower_bounds[1].grid(row=2,column=2)
        self.entry_upper_bounds.append(tk.Entry(self, textvariable=self.upper_bounds[1]))
        self.entry_upper_bounds[1].grid(row=2,column=3)
        
        for i,name in enumerate(fitmodel.model.parameter_names, start=2):
            
            # add labels for fitting parameter names
            self.label.append(tk.Label(self, text=name, justify='left'))
            self.label[i].grid(row=i+1,column=0)
            
            # initialize string variables for the fitting parameters
            self.values.append(tk.StringVar())
            self.values[i].set(str(fitmodel.model.K[i-2]))
            self.lower_bounds.append(tk.StringVar())
            self.lower_bounds[i].set(str(fitmodel.lower_bounds[i]))
            self.upper_bounds.append(tk.StringVar())
            self.upper_bounds[i].set(str(fitmodel.upper_bounds[i]))
            
            # add entry boxes for fitting parameters entry
            self.entry_values.append(tk.Entry(self, textvariable=self.values[i]))
            self.entry_values[i].grid(row=i+1,column=1)
            self.entry_lower_bounds.append(tk.Entry(self, textvariable=self.lower_bounds[i]))
            self.entry_lower_bounds[i].grid(row=i+1,column=2)
            self.entry_upper_bounds.append(tk.Entry(self, textvariable=self.upper_bounds[i]))
            self.entry_upper_bounds[i].grid(row=i+1,column=3)
        
        # add buttons for user confirmation or cancellation
        self.ok_button = tk.Button(self, text='Ok', command=self.destroy, width=15)
        self.ok_button.grid(row=len(fitmodel.model.parameter_names)+3,column=1,columnspan=2)
        
        # make this window modal
        self.wait_window()
        
    def get_initial_guess(self):
        
        values = []
        lower_bounds = []
        upper_bounds = []
        for i in range(len(self.values)):
            values.append(float(self.values[i].get()))
            lower_bounds.append(float(self.lower_bounds[i].get()))
            upper_bounds.append(float(self.upper_bounds[i].get()))
        
        return values, lower_bounds, upper_bounds
    
#####################################################################################################################
#####################################################################################################################      

class MCRGui(tk.Toplevel):
    
    def __init__(self, master, TA, fitmodel):
        
        super().__init__(master) # parent class (=tk.Toplevel) initialization
        
        self.toolbar_frame = tk.Frame(self)
        
        self.TA = TA
        
        self.fitmodel = fitmodel
        
        # create instructions label
        self.instructions_label = ttk.Label(self,
                                            justify='center',
                                            text = 'Select delay times for spectral initial guess. Separate by spaces'
                                            )
        
        # create matplotlib figure
        self.fig = plt.Figure(figsize=(11,5), dpi=100)
        self.ax_spectra = self.fig.add_subplot(121)
        self.ax_populations = self.fig.add_subplot(122)
        
        # create figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create toolbar. The toolbar only sits in a frame because we need to pack it using the grid manager (toolbar by itself can only use pack manager)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack()
        
        # create frame to hold all user input widgets
        self.user_input_frame = tk.Frame(self)
        
        # add label to entry widget
        self.delays_label = tk.Label(self.user_input_frame, text='Selected delays')
        
        # add string var that holds selected delay times for initial guess
        self.delays = tk.StringVar()
        self.delays.set(' '.join([str(int(x)) for x in np.linspace(1,max(self.TA.delay),len(self.fitmodel.model.initial_populations))]))
        
        # create entry box for delay times selection
        self.delays_entry = tk.Entry(self.user_input_frame, textvariable=self.delays)
        
        # create button widgets
        self.accept_fit_button = ttk.Button(self.user_input_frame, text='Accept fit', command=self.accept_fit)
        self.MCR_fit_button = ttk.Button(self.user_input_frame, text='MCR-ALS fit', command=self.mcrals_fit)
        
        # pack the widgets onto the frames
        self.instructions_label.grid(row=0, column=0)
        self.canvas.get_tk_widget().grid(row=1, column=0)
        self.toolbar_frame.grid(row=2, column=0)
        self.user_input_frame.grid(row=3, column=0)
        self.delays_label.grid(row=0, column=0, sticky='e')
        self.delays_entry.grid(row=0, column=1, sticky='w')
        self.MCR_fit_button.grid(row=1, column=0, columnspan=2)
        self.accept_fit_button.grid(row=2, column=0, columnspan=2)
        
        # make this window modal
        self.wait_window()
        
    def accept_fit(self):
        
        self.destroy()
    
    def mcrals_fit(self):
        
        # perform MCR-ALS fitting
        try:
            time_points = [float(x) for x in self.delays.get().split()]
        except ValueError:
            print('ERROR: please enter delay times separated by spaces')
            return
        
        # check that user supplied number of points equal to the number of species in the model
        if len(time_points)!=len(self.fitmodel.model.initial_populations):
            tk.messagebox.showerror('Wrong number of time points provided', 'Please enter {0} time point(s) for {0} species'.format(str(len(self.fitmodel.model.initial_populations))))
            return
        
        time_points_inds = [np.argmin(np.abs(self.TA.delay-tp)) for tp in time_points]
        initial_spectra = self.TA.deltaA[:,time_points_inds]
        self.fitmodel.mcrals.fit(self.TA.deltaA.T, ST=initial_spectra.T, st_fix=[], c_fix=[], verbose=False)
        
        # normalize MCR-ALS populations
        mcrals_populations = self.fitmodel.mcrals.C_opt_
        for i in range(mcrals_populations.shape[1]):
            for i in range(mcrals_populations.shape[1]):
                mcrals_populations[:,i] = mcrals_populations[:,i]/np.max(np.abs(mcrals_populations[:,i]))
        
        # plot result
        self.plot_figure()
        
    def get_fitmodel(self):
        
        return self.fitmodel
    
    def plot_figure(self):
        
        if self.TA == []:
            return
        
        self.ax_spectra.cla()
        self.ax_populations.cla()
        
        self.ax_spectra.plot([min(self.TA.wavelength), max(self.TA.wavelength)], [0,0], color='black', alpha=0.5)
        self.ax_populations.plot([min(self.TA.delay), max(self.TA.delay)], [0,0], color='black', alpha=0.5)
        
        for i in range(self.fitmodel.mcrals.n_targets):
            self.ax_spectra.plot(self.TA.wavelength, self.fitmodel.mcrals.ST_opt_[i,:]/np.max(np.abs(self.fitmodel.mcrals.ST_opt_[i,:])))
            self.ax_populations.plot(self.TA.delay, self.fitmodel.mcrals.C_opt_[:,i])
        
        self.ax_spectra.set_xlabel('Wavelength')
        self.ax_spectra.set_ylabel('Norm. species spectra')
        self.ax_spectra.set_ylim((-1.1, 1.1))
        self.ax_spectra.set_xlim((min(self.TA.wavelength), max(self.TA.wavelength)))
        
        self.ax_populations.set_xscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax_populations.set_xlabel('Time')
        self.ax_populations.set_ylabel('Norm. populations')
        self.ax_populations.set_ylim((-0.1, 1.1))
        self.ax_populations.set_xlim((min(self.TA.delay), max(self.TA.delay)))
        
        self.ax_spectra.figure.canvas.draw()
        self.ax_populations.figure.canvas.draw()
        self.toolbar.update()
