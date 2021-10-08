import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm # color maps
from matplotlib.colors import to_hex, to_rgb

class PlotAreaFrame(tk.Frame):
    
    def __init__(self, master):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        # initialize sub-frames to contain figures and toolbars
        self.frame_fig1 = tk.Frame(self)  
        self.frame_fig2 = tk.Frame(self)
        self.frame_fig3 = tk.Frame(self)
        self.frame_POI = tk.Frame(self)
                
        # initialize the matplotlib figures (size, dpi, position, axes labels, etc...)
        self.init_figures()
        
        # make Tk/Matplotlib figure canvases on the plot area frame and draw onto them the plots generated previously
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.frame_fig1)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame_fig2)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.frame_fig3)
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        
        # create toolbars
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.frame_fig1)
        self.toolbar1.update()
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.frame_fig2)
        self.toolbar2.update()
        self.toolbar3 = NavigationToolbar2Tk(self.canvas3, self.frame_fig3)
        self.toolbar3.update()
        
        # create tree widget to hold POIs
        self.POI_table = POITable(self.frame_POI)
        
        # create scrollbar for POI table
        self.POI_table_scrollbar = ttk.Scrollbar(master=self.frame_POI,
                                                 orient='vertical',
                                                 command=self.POI_table.yview
                                                 )
        self.POI_table.config(yscrollcommand=self.POI_table_scrollbar.set)
        
        # create header label for POI table
        self.POI_label = ttk.Label(self.frame_POI,text='Points of interest')
        
        # use geometry managers to pack the widgets within each frame
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar1.pack()
        self.toolbar2.pack()
        self.toolbar3.pack()
        self.POI_label.grid(row=0,column=0,sticky='n')
        self.POI_table.grid(row=1,column=0,sticky='ns')
        self.POI_table_scrollbar.grid(row=1,column=1,sticky='ns')
        
        # Use the grid geometry manager to pack the widgets in the main plot area frame
        self.frame_fig1.grid(row=0,column=0,sticky='nsew')
        self.frame_fig2.grid(row=0,column=1,sticky='nsew')
        self.frame_fig3.grid(row=1,column=0,sticky='nsew')
        self.frame_POI.grid(row=1,column=1,sticky='ns')    
        
    def init_figures(self):
        
        # create figure 1
        self.fig1 = plt.Figure(figsize=(5,3.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_position([0.18,0.17,0.77,0.78])
        self.fig1.set_tight_layout(True)
        
        # create figure 2
        self.fig2 = plt.Figure(figsize=(5,3.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_position([0.18,0.17,0.77,0.78])
        self.fig2.set_tight_layout(True)
        
        # create figure 3
        self.fig3 = plt.Figure(figsize=(5,3.5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_position([0.18,0.17,0.77,0.78])
        self.fig3.set_tight_layout(True)
        
    def refresh_fig1(self, TA, fitmodel=[]):
        
        # start by plotting the TA surface
        if TA == []:
            return
        
        self.ax1.cla()
        
        if fitmodel != []: # plot the experimental and fitted kinetic traces overlaid
            
            # get fitting parameters from the fitmodel object
            fit_params = np.concatenate(([fitmodel.irf], fitmodel.model.K))
            deltaA_residuals = fitmodel.calc_model_deltaA_residual_matrix(fit_params).transpose()
            scaled_deltaA_residuals = -np.arctan(deltaA_residuals*1000)
            self.ax1.pcolormesh(TA.wavelength, TA.delay, scaled_deltaA_residuals, shading='nearest', cmap=cm.RdBu)
            self.ax1.set_title(r'$\Delta$A residuals')
            
        else:     
            
            deltaA = np.flipud(TA.deltaA).transpose()
            scaledTA = -np.arctan(deltaA*1000)
            self.ax1.pcolormesh(TA.wavelength[::-1], TA.delay, scaledTA, shading='nearest', cmap=cm.RdBu)
            self.ax1.set_title(r'$\Delta$A surface')
            
        # add the points of interest
        for iid in self.POI_table.get_children():
            wavelength, delay, color = self.POI_table.get_poi(iid)
            self.ax1.plot(wavelength, delay, marker='o', markeredgecolor=color, markerfacecolor=color)

        self.ax1.set_yscale('symlog', linthresh=1.0, linscale=0.35)
        xlims = self.ax1.get_xlim()
        ylims = self.ax1.get_ylim()
        self.ax1.set_xlim(xlims)
        self.ax1.set_ylim(ylims)
        self.ax1.set_xlabel('Wavelength')
        self.ax1.set_ylabel('Delay')
        
        # show plot on GUI
        self.ax1.figure.canvas.draw()
        self.toolbar1.update()
        self.toolbar2.update()
        self.toolbar3.update()
    
    def refresh_fig2(self, TA, fitmodel=[]):
        
        if TA == []:
            return
        
        self.ax2.cla()
        
        if fitmodel != []: # plot the experimental and fitted kinetic traces overlaid
            
            # get fitting parameters from the fitmodel object
            fit_params = np.concatenate(([fitmodel.irf], fitmodel.model.K))

            for i,iid in enumerate(self.POI_table.get_children()):
                wavelength, delay, color = self.POI_table.get_poi(iid)
                wavelength_index = np.absolute(wavelength-TA.wavelength).argmin()
                self.ax2.plot(TA.delay, TA.deltaA[wavelength_index,:]*1000, color=color, marker='o', linestyle='none', markersize=3, alpha=0.25, markeredgecolor='none')
                self.ax2.plot(TA.delay, fitmodel.calc_model_kinetic_traces(fit_params)[:,i]*1000, color=color, lw=1)
                
        else: # plot only the experimental kinetic traces
            
            for iid in self.POI_table.get_children():
                wavelength, delay, color = self.POI_table.get_poi(iid)
                wavelength_index = np.absolute(wavelength-TA.wavelength).argmin()
                self.ax2.plot(TA.delay, TA.deltaA[wavelength_index,:]*1000, lw=1, color=color, alpha=0.75)
                
        self.ax2.set_xscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax2.set_xlim((min(TA.delay),max(TA.delay)))
                
        # add line at deltaA=0
        self.ax2.plot([min(TA.delay), max(TA.delay)],[0,0], linewidth=0.3, color=(0.2,0.2,0.2))
        
        # update axis labels
        self.ax2.set_xlabel('Delay')
        self.ax2.set_ylabel(r'$\Delta$A $\times 10^3$')
        
        # show plot on GUI
        self.ax2.figure.canvas.draw()
        
    def refresh_fig3(self, TA, fitmodel=[]):
        
        if TA == []:
            return
        
        self.ax3.cla()
        
        if fitmodel != []: # plot the species associated spectra
            
            # get fitting parameters from the fitmodel object
            fit_params = np.concatenate(([fitmodel.irf], fitmodel.model.K))
            
            # calculate normalized species associated spectra
            model_species_spectra = fitmodel.calc_model_species_spectra(fit_params)
            model_species_spectra = fitmodel.normalize_species_spectra(model_species_spectra)
   
            # plot the species spectra
            for i in range(model_species_spectra.shape[1]):
                self.ax3.plot(TA.wavelength, model_species_spectra[:,i], label='species '+str(i+1))
                self.ax3.set_ylabel('Normalized SAS')
                
            # update legend
            self.ax3.legend()
        
        else: # plot the spectra of the selected wavelengths (in POIs)
            
            for iid in self.POI_table.get_children():
                wavelength, delay, color = self.POI_table.get_poi(iid)
                delay_index = np.argmin(abs(TA.delay-delay))
                self.ax3.plot(TA.wavelength, TA.deltaA[:,delay_index]*1000, lw=1, color=color, alpha=0.75)
                self.ax3.set_ylabel(r'$\Delta$A $\times 10^3$')
        
        # update axis labels
        self.ax3.set_xlabel('Wavelength')
        
        # add line at deltaA=0
        self.ax3.plot([min(TA.wavelength), max(TA.wavelength)],[0,0], linewidth=0.3, color=(0.2,0.2,0.2))
        
        # match wavelength axis limits to the limits of figure 1
        self.ax3.set_xlim(self.ax1.get_xlim())
        
        # show plot on GUI
        self.ax3.figure.canvas.draw()
        
#####################################################################################################################
#####################################################################################################################

class POITable(ttk.Treeview):
    
    def __init__(self, master):
        
        # initialize treeview
        super().__init__(master,
                         columns=('col#1', 'col#2'),
                         show='headings',
                         height=11,
                         selectmode = 'browse'
                         )
        
        self.num_POIs = 0
        
        # fix problem of style overiding row background set according to tag
        self.style = ttk.Style()
        self.style.map("Treeview",
                       foreground=self.fixed_map("foreground"),
                       background=self.fixed_map("background"))
        
        # set columns text and appearance
        self.heading('col#1', text='Wavelength')
        self.heading('col#2', text='Delay')
        self.column('col#1', width=100)
        self.column('col#2', width=100)
        
        # setup colors for POIs
        self.setup_colors(num_colors=15)
        
    def insert_poi(self, wavelength, delay):
        
        # set text values that will appear in the columns of the table
        formatted_values = (
                            "{:.3f}".format(wavelength),
                            "{:.3f}".format(delay)
                            )
        
        # set the color tag for that POI
        color = self.get_next_color()
        self.insert('', 0, values=formatted_values, tags=color)
        
        self.num_POIs = self.num_POIs + 1
        
    def remove_poi(self, iid):
        
        self.delete(iid)
        
        self.num_POIs = self.num_POIs - 1
    
    def clear_table(self):
        
        for iid in self.get_children(): self.remove_poi(iid)
        
        self.num_POIs = 0
    
    def get_poi(self, iid):
        
        wavelength = float(self.item(iid)['values'][0])
        delay = float(self.item(iid)['values'][1])
        color = to_rgb(self.item(iid)['tags'][0])
        return (wavelength, delay, color)
    
    def get_all_wavelengths(self):
        
        wavelengths = []
        for iid in self.get_children():
            wavelengths.append(float(self.item(iid)['values'][0]))
        
        return wavelengths
    
    def get_all_delays(self):
        
        delays = []
        for iid in self.get_children():
            delays.append(float(self.item(iid)['values'][1]))
        
        return delays
    
    def get_next_color(self):
        
        # save the next color
        next_color = self.colors[self.color_indices[0]]
        
        # roll over the color indices so that the next time a new color is selected
        self.color_indices = np.roll(self.color_indices, shift=-1)
        
        # return the color
        return next_color
    
    def setup_colors(self, num_colors=5):
        
        # set color list (in hex format) and color_indices list that will cyclicly roll when a new point is added
        cm_selector = np.linspace(0.9, 0.1, num_colors)
        self.colors = [to_hex(cm.turbo(x)) for x in cm_selector]
        self.color_indices = np.arange(num_colors) # the first index (=0) is the always the next color
        
        # associate color tags with row color
        for color in self.colors:
            self.tag_configure(tagname=color, background=color)
            
    def fixed_map(self, option):
        # function I found on the internet that helps to show colored rows in Treeview. If not for this function, the rows will ot be colored
        return [elm for elm in self.style.map("Treeview", query_opt=option)
                if elm[:2] != ("!disabled", "!selected")]