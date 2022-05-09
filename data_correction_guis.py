import tkinter as tk
from tkinter import ttk

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm # color maps
import matplotlib.colors as colors

#####################################################################################################################
#####################################################################################################################

class ChirpCorrectGui(tk.Toplevel):
    
    def __init__(self, master, TA):
        
        super().__init__(master) # parent class (=tk.Toplevel) initialization
        
        self.toolbar_frame = tk.Frame(self)
        
        self.TA = TA
        
        # create instructions label
        self.instructions_label = ttk.Label(self,
                                            justify='center',
                                            text = 'Click on TA surface to insert time-zero points at different wavelengths.\nWhen more than 4 points are inserted, a chirp line will be displayed.'
                                            )
        
        # create matplotlib figure
        self.fig = plt.Figure(figsize=(7,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.15,0.17,0.80,0.78])
        
        # create figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create toolbar. The toolbar only sits in a frame because we need to pack it using the grid manager (toolbar by itself can only use pack manager)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack()
        
        # create tree widget
        self.point_list = ttk.Treeview(master=self,
                                     columns=('col#1', 'col#2'),
                                     show='headings',
                                     height=10,
                                     selectmode = 'browse'
                                     )
        self.point_list.heading('col#1', text='Wavelength (nm)')
        self.point_list.heading('col#2', text='t0 (ps)')
        self.point_list.column('col#1', width=100)
        self.point_list.column('col#2', width=100)
        
        # create button widgets
        self.accept_fit_button = ttk.Button(self, text='Accept fit', command=self.accept_fit)
        self.clear_points_button = ttk.Button(self, text='Clear all points', command=self.clear_all_points)
        
        # initialize the chirp (delay vs. wavelength)
        self.calculate_chirp()
        
        # plot TA surface
        self.plot_figure()
        
        # pack the widgets onto the frames
        self.instructions_label.grid(row=0, column=0, columnspan=2)
        self.canvas.get_tk_widget().grid(row=1, column=0)
        self.toolbar_frame.grid(row=2, column=0, sticky='w')
        self.point_list.grid(row=1, column=1, rowspan=2, sticky='n')
        self.accept_fit_button.grid(row=3, column=0)
        self.clear_points_button.grid(row=3, column=1)
        
        # event bindings
        self.event_add_point = self.fig.canvas.mpl_connect('button_press_event', self.add_point)
        
        # make this window modal
        self.wait_window()
        
    def plot_figure(self):
        
        if self.TA == []:
            return
        
        # plot TA surface
        deltaA = np.flipud(self.TA.deltaA).transpose()
        scaledTA = np.arctan(deltaA/np.max(deltaA)) # just for plotting! normalize deltaA and use arctan transformation
        self.ax.cla()
        # self.ax.pcolormesh(self.TA.wavelength[::-1], self.TA.delay, scaledTA, shading='nearest', cmap=cm.twilight)
        # self.ax.contour(self.TA.wavelength[::-1], self.TA.delay, scaledTA, colors='white', alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10))
        self.ax.contourf(self.TA.wavelength[::-1], self.TA.delay, scaledTA, cmap=cm.twilight, levels=40)
        
        # plot sampled chirp points
        for iid in self.point_list.get_children():
            wavelength = float(self.point_list.item(iid)['values'][0])
            delay = float(self.point_list.item(iid)['values'][1])
            self.ax.plot(wavelength, delay, marker='o', markerfacecolor='blue', markeredgecolor='blue')
        
        # plot chirp
        self.ax.plot(self.TA.wavelength, self.chirp, color='blue')
        
        self.ax.set_yscale('symlog', linthresh=10.0, linscale=2.0)
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Time')
        self.ax.figure.canvas.draw()
        self.toolbar.update()
        
    def add_point(self, event):
        
        # if the toolbar is in either 'zoom' or 'pan' mode, do not add point
        if (self.toolbar.mode != 'zoom rect') and (self.toolbar.mode != 'pan/zoom'):
            wavelength, delay = event.xdata, event.ydata
            self.point_list.insert('', tk.END, values=("{:.2f}".format(wavelength), "{:.2f}".format(delay)))
            self.calculate_chirp()
            self.plot_figure()
            
    def calculate_chirp(self):
        
        self.chirp = np.zeros(self.TA.wavelength.shape)
        self.chirp_coeffs = np.array([0.,0.,0.])
        if len(self.point_list.get_children()) >= 4:
            sampled_wavelengths = [float(self.point_list.item(iid)['values'][0]) for iid in self.point_list.get_children()]
            sampled_delays = [float(self.point_list.item(iid)['values'][1]) for iid in self.point_list.get_children()]
            self.chirp_coeffs, pcov = scipy.optimize.curve_fit(self.TA.chirp_fit_func, sampled_wavelengths, sampled_delays)
            self.chirp = self.TA.chirp_fit_func(self.TA.wavelength, *self.chirp_coeffs)
        
    def clear_all_points(self):
        
        for iid in self.point_list.get_children():
            self.point_list.delete(iid)
            
        self.calculate_chirp()
        self.plot_figure()
        
    def accept_fit(self):
              
        # run a final calculation of the chirp coefficients
        self.calculate_chirp()
        
        # close the window
        self.destroy()
    
    def get_chirp_coeffs(self):
        
        return self.chirp_coeffs
        
#####################################################################################################################
#####################################################################################################################

class SubtractBackgroundGui(tk.Toplevel):
    
    def __init__(self, master, TA):
        
        super().__init__(master)  # parent class (=tk.Toplevel) initialization
        
        self.TA = TA
        
        # create matplotlib figure
        self.fig = plt.Figure(figsize=(8,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_position([0.15,0.17,0.80,0.78])
                
        # create figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create spinbox that shows how many spectra are averaged
        self.num_spectra_to_subtract = tk.StringVar(value=0)
        self.spinbox = ttk.Spinbox(master=self,
                                     from_=0,
                                     to=self.TA.delay.size,
                                     textvariable=self.num_spectra_to_subtract,
                                     wrap=True,
                                     command=self.calc_pret0_background
                                     )
        
        # create button widgets
        self.accept_button = ttk.Button(self, text='Accept', command=self.accept)
        
        # pack the widgets
        self.canvas.get_tk_widget().pack()
        self.spinbox.pack()
        self.accept_button.pack()
        
        # calculate pre-time-zero background
        self.calc_pret0_background()
        
        # plot on canvas
        self.update_figure()
        
        # make this window modal
        self.wait_window()
        
    def calc_pret0_background(self):
        
        # if user selected zero spectra, the pre-t0 background is assumed to be 0
        if int(self.num_spectra_to_subtract.get())==0:
            self.pret0_background = np.zeros(self.TA.wavelength.shape)
        
        # calculate pre-t0 background only if averaging more than 1 curve
        elif int(self.num_spectra_to_subtract.get())==1:
            self.pret0_background = self.TA.deltaA[:,1]
        
        # user selected only one spectra as pre-t0 background
        else:
            self.pret0_background = np.mean(self.TA.deltaA[:,1:int(self.num_spectra_to_subtract.get())], axis=1)
        
        # finish by updating the figure
        self.update_figure()
        
    def update_figure(self):
        
        # clear axes
        self.ax.cla()
        
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel(r'$\Delta$A $\times 10^3$')
        
        for i in range(int(self.num_spectra_to_subtract.get())):
            self.ax.plot(self.TA.wavelength, self.TA.deltaA[:,i]*1000, color='black', alpha=0.1)
        self.ax.plot(self.TA.wavelength, self.pret0_background*1000, color='red')
        
        # update plot
        self.ax.figure.canvas.draw()
    
    def accept(self):
        
        # calculate pre-t0 background one last time before closing window
        self.calc_pret0_background()
        
        # close the window
        self.destroy()
        
    def get_pret0_background(self):
        
        return self.pret0_background

#####################################################################################################################
#####################################################################################################################

class SVDGui(tk.Toplevel):
    
    def __init__(self, master, TA):
        
        super().__init__(master) # parent class (=tk.Toplevel) initialization
        
        self.toolbar_frame = tk.Frame(self)
        self.component_selection_frame = tk.Frame(self)
        
        # get TA data and perform SVD on it
        self.TA = TA
        self.U, self.S, self.VH = np.linalg.svd(self.TA.deltaA)
        
        # create matplotlib figures
        self.fig = plt.Figure(figsize=(9,6))
        self.fig.set_tight_layout(True)
        
        # create axes
        self.ax1 = self.fig.add_subplot(2,2,1)
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)
        
        self.residuals_colorbar = self.fig.colorbar(self.ax4.pcolormesh((0,1),(0,1),((0,0),(0,0)), shading='auto'), ax=self.ax4, label='mOD')
        
        # create figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create toolbar. The toolbar only sits in a frame because we need to pack it using the grid manager (toolbar by itself can only use pack manager)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack()
        
        # create component selection checkbuttons and variables
        self.N_components = 9 # max number of SVD components
        self.component_variables = [tk.IntVar(self.component_selection_frame) for i in range(self.N_components)]
        self.component_checkbuttons = [tk.Checkbutton(self.component_selection_frame, text='Comp. '+str(i+1), variable=self.component_variables[i]) for i in range(self.N_components)]
        
        # pack checkbuttons to frame
        for i in range(self.N_components):
            
            # pack the checkbuttons
            Nrows=3 # number of rows
            self.component_checkbuttons[i].grid(row=np.mod(i,Nrows),column=int(np.floor(i/Nrows)))
            
            # trace changes in the intvars
            self.component_variables[i].trace_add('write', self.plot_figure)
            
        # create button widget
        self.apply_SVD_filter_button = ttk.Button(self, text='Apply SVD filter', command=self.apply_SVD_filter)
        
        # # plot singular values
        # self.ax4.cla()
        # self.ax4.plot(np.array(range(self.N_components))+1, self.S[:self.N_components], marker='o')
        # self.ax4.set_yscale('log')
        # self.ax4.set_xlabel('Comp. #')
        # self.ax4.set_ylabel('S')
        # self.ax4.figure.canvas.draw()
        
        # pack the widgets onto the frames
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.toolbar_frame.grid(row=1,column=0, sticky='w')
        self.component_selection_frame.grid(row=2,column=0)
        self.apply_SVD_filter_button.grid(row=3,column=0)
        
        # make this window modal
        self.wait_window()
    
    def get_selected_SVD_indices(self):
        
        # outputs a vector of the SVD component numbers based on the checkbox selection
        variable_list = [x.get() for x in self.component_variables]
        SVD_indices = []
        for i in range(len(variable_list)):
            if variable_list[i]==1:
                SVD_indices.append(i)  
        return SVD_indices
        
    def get_SVD_deltaA_components(self, selected_SVD_indices):
        
        SVD_components = np.zeros(self.TA.deltaA.shape)
        for i in selected_SVD_indices:
            SVD_components = SVD_components + self.S[i] * np.array((self.U[:,i],)).T @ np.array((self.VH[i,:],))
        return SVD_components
    
    def get_SVD_kinetic_components(self, selected_SVD_indices):
        
        SVD_components = np.zeros((len(self.TA.delay),len(selected_SVD_indices)))
        for i,n in enumerate(selected_SVD_indices):
            SVD_components[:,i] = self.S[n] * np.array(self.VH[n,:]).T
        
        return SVD_components
    
    def get_SVD_spectrum_components(self, selected_SVD_indices):
        
        SVD_components = np.zeros((len(self.TA.wavelength),len(selected_SVD_indices)))
        for i,n in enumerate(selected_SVD_indices):
            SVD_components[:,i] = self.S[n] * np.array(self.U[:,n])
        
        return SVD_components
    
    def plot_figure(self, *args):
        
        selected_SVD_indices = self.get_selected_SVD_indices()
        if selected_SVD_indices==[]:
            return
        
        # plot selected components on deltaA axes
        SVD_deltaA = np.flipud(self.get_SVD_deltaA_components(selected_SVD_indices)).transpose()
        scaled_SVD_deltaA = -np.arctan(SVD_deltaA*1000)
        self.ax1.cla()
        # self.ax1.pcolormesh(self.TA.wavelength[::-1], self.TA.delay, scaled_SVD_deltaA, shading='nearest', cmap=cm.RdBu)
        self.ax1.contourf(self.TA.wavelength[::-1], self.TA.delay, scaled_SVD_deltaA, cmap=cm.RdBu, levels=40)
        self.ax1.set_title(r'Filtered $\Delta$A surface')
        self.ax1.set_yscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax1.set_xlabel('Wavelength')
        self.ax1.set_ylabel('Time')
        self.ax1.figure.canvas.draw()
        
        # plot selected component kinetics
        self.ax2.cla()
        self.ax2.plot(self.TA.delay, self.get_SVD_kinetic_components(selected_SVD_indices))
        self.ax2.plot([min(self.TA.delay),max(self.TA.delay)], [0,0], color='black', alpha=0.5)
        self.ax2.set_xscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax2.set_xlim((min(self.TA.delay),max(self.TA.delay)))
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel(r'$\Delta$A')
        self.ax2.figure.canvas.draw()
        
        # plot selected component spectra
        self.ax3.cla()
        self.ax3.plot(self.TA.wavelength, self.get_SVD_spectrum_components(selected_SVD_indices))
        self.ax3.plot([min(self.TA.wavelength),max(self.TA.wavelength)], [0,0], color='black', alpha=0.5)
        self.ax3.set_xlim((min(self.TA.wavelength),max(self.TA.wavelength)))
        self.ax3.set_xlabel('Wavelength')
        self.ax3.set_ylabel(r'$\Delta$A')
        self.ax3.figure.canvas.draw()
        
        # plot residual matrix
        SVD_deltaA = self.get_SVD_filtered_deltaA()[1]
        residual_matrix = (self.TA.deltaA - SVD_deltaA).T
        self.ax4.cla()
        self.residuals_plot = self.ax4.contourf(self.TA.wavelength[::-1], self.TA.delay, residual_matrix, cmap=cm.RdBu, levels=40, norm=colors.CenteredNorm())
        self.ax4.set_title(r'$\Delta$A residuals')
        self.residuals_colorbar.update_normal(self.residuals_plot)
        self.ax4.set_yscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax4.set_xlabel('Wavelength')
        self.ax4.set_ylabel('Time')
        self.ax4.figure.canvas.draw()
        
        self.toolbar.update()
        
    def get_SVD_filtered_deltaA(self):

        # return the SVD components used and the filtered deltaA based on the selected components
        return self.get_selected_SVD_indices(), self.get_SVD_deltaA_components(self.get_selected_SVD_indices())
    
    def apply_SVD_filter(self):
        
        # close the window
        self.destroy()