import tkinter as tk
from tkinter import ttk

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm # color maps

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
        scaledTA = -np.arctan(deltaA*1000)
        self.ax.cla()
        self.ax.pcolormesh(self.TA.wavelength[::-1], self.TA.delay, scaledTA, shading='nearest', cmap=cm.RdBu)
        
        # plot sampled chirp points
        for iid in self.point_list.get_children():
            wavelength = float(self.point_list.item(iid)['values'][0])
            delay = float(self.point_list.item(iid)['values'][1])
            self.ax.plot(wavelength, delay, marker='o', markerfacecolor='blue', markeredgecolor='blue')
        
        # plot chirp
        self.ax.plot(self.TA.wavelength, self.chirp, color='blue')
        
        self.ax.set_yscale('symlog', linthresh=1.0, linscale=0.35)
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