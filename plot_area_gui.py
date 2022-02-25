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
        
        # initialize frame to contain POIs
        self.frame_POI = tk.Frame(self)
        
        # create plot toolbar frame
        self.toolbar_frame = tk.Frame(self)
        
        # create the figure
        self.fig = plt.Figure(figsize=(8,6), dpi=100)
        self.fig.set_tight_layout(True)
        
        # create axes
        self.ax1 = self.fig.add_subplot(2,2,1)
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)
        
        # make Tk/Matplotlib figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create toolbars
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack()
    
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
        
        # use geometry managers to pack the widgets
        self.POI_label.grid(row=0,column=0,sticky='n')
        self.POI_table.grid(row=1,column=0,sticky='ns')
        self.POI_table_scrollbar.grid(row=1,column=1,sticky='ns')
        
        # Use the grid geometry manager to pack the widgets in the main plot area frame
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.frame_POI.grid(row=0,column=1,sticky='ns')
        self.toolbar_frame.grid(row=1,column=0,sticky='w')
        
    def refresh_fig1(self, TA, fitmodel=[]):
        
        # start by plotting the TA surface
        if TA == []:
            return
        
        self.ax1.cla()
        
        deltaA = np.flipud(TA.deltaA).transpose()
        scaledTA = np.arctan(deltaA/np.max(deltaA)) # just for plotting! normalize deltaA and use arctan transformation
        self.ax1.pcolormesh(TA.wavelength[::-1], TA.delay, scaledTA, shading='nearest', cmap=cm.twilight)
        self.ax1.contour(TA.wavelength[::-1], TA.delay, scaledTA, colors='white', alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10))
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
        self.toolbar.update()
    
    def refresh_fig2(self, TA, fitmodel=[]):
        
        if TA == []:
            return
        
        self.ax2.cla()
        
        if fitmodel != []: # plot the experimental and fitted kinetic traces overlaid
            
            # get fitting parameters from the fitmodel object
            fit_params = np.concatenate(([fitmodel.irf, fitmodel.tzero], fitmodel.model.K))

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
            fit_params = np.concatenate(([fitmodel.irf, fitmodel.tzero], fitmodel.model.K))
            
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
    
    def refresh_fig4(self, TA, fitmodel=[]):
        
        # start by plotting the TA surface
        if TA == []:
            return
        
        self.ax4.cla()
        
        if fitmodel != []: # plot the experimental and fitted kinetic traces overlaid
            
            # get fitting parameters from the fitmodel object
            fit_params = np.concatenate(([fitmodel.irf, fitmodel.tzero], fitmodel.model.K))
            deltaA_residuals = fitmodel.calc_model_deltaA_residual_matrix(fit_params).transpose()
            scaled_deltaA_residuals = -np.arctan(deltaA_residuals*1000)
            self.ax4.pcolormesh(TA.wavelength, TA.delay, scaled_deltaA_residuals, shading='nearest', cmap=cm.RdBu)
            self.ax4.set_title(r'$\Delta$A residuals')
            
            # add the points of interest
            for iid in self.POI_table.get_children():
                wavelength, delay, color = self.POI_table.get_poi(iid)
                self.ax4.plot(wavelength, delay, marker='o', markeredgecolor=color, markerfacecolor=color)

        self.ax4.set_yscale('symlog', linthresh=1.0, linscale=0.35)
        xlims = self.ax4.get_xlim()
        ylims = self.ax4.get_ylim()
        self.ax4.set_xlim(xlims)
        self.ax4.set_ylim(ylims)
        self.ax4.set_xlabel('Wavelength')
        self.ax4.set_ylabel('Delay')
        
        # show plot on GUI
        self.ax4.figure.canvas.draw()
        
    def plot_compared_kinetics(self, TA, TA_compare):
        
        plt.figure()
        ax_kinetic_compare = plt.axes()        
        
        for n,iid in enumerate(self.POI_table.get_children()):
            wavelength, delay, color = self.POI_table.get_poi(iid)
            wavelength_index = np.absolute(wavelength-TA.wavelength).argmin()
            wavelength_index_compare = np.absolute(wavelength-TA_compare.wavelength).argmin()
            ax_kinetic_compare.plot(TA.delay, TA.deltaA[wavelength_index,:]/np.max(np.abs(TA.deltaA[wavelength_index,:])), lw=3, alpha=0.5, color=plt.cm.tab20(2*n), label=str(wavelength)+' nm, original')
            ax_kinetic_compare.plot(TA_compare.delay, TA_compare.deltaA[wavelength_index_compare,:]/np.max(np.abs(TA_compare.deltaA[wavelength_index_compare,:])), lw=3, alpha=0.5, color=plt.cm.tab20(2*n+1), label=str(wavelength)+' nm, compared')
            
        ax_kinetic_compare.set_xscale('symlog', linthresh=1.0, linscale=0.35)
        ax_kinetic_compare.set_xlim((min(TA.delay),max(TA.delay)))
                
        # add line at deltaA=0
        ax_kinetic_compare.plot([min(TA.delay), max(TA.delay)],[0,0], linewidth=0.3, color=(0.2,0.2,0.2))
        
        # update axis labels
        ax_kinetic_compare.set_xlabel('Delay')
        ax_kinetic_compare.set_ylabel(r'Norm. $\Delta$A')
        ax_kinetic_compare.legend()
        
    def plot_compared_spectra(self, TA, TA_compare):
        
        plt.figure()
        ax_spectrum_compare = plt.axes()
        
        
        for n,iid in enumerate(self.POI_table.get_children()):
            wavelength, delay, color = self.POI_table.get_poi(iid)
            delay_index = np.absolute(delay-TA.delay).argmin()
            delay_index_compare = np.absolute(delay-TA_compare.delay).argmin()
            ax_spectrum_compare.plot(TA.wavelength, TA.deltaA[:,delay_index], lw=3, alpha=0.5, color=plt.cm.tab20(2*n), label=str(delay)+' ps, original')
            ax_spectrum_compare.plot(TA_compare.wavelength, TA_compare.deltaA[:,delay_index_compare], lw=3, alpha=0.5, color=plt.cm.tab20(2*n+1), label=str(delay)+' ps, compared')
                
        # add line at deltaA=0
        ax_spectrum_compare.plot([min(TA.wavelength), max(TA.wavelength)],[0,0], linewidth=0.3, color=(0.2,0.2,0.2))
        
        # update axis labels
        ax_spectrum_compare.set_xlabel('Wavelength')
        ax_spectrum_compare.set_ylabel(r'$\Delta$A $\times 10^3$')
        ax_spectrum_compare.legend()
        
    def plot_compared_surfaces(self, TA, TA_compare):
        
        plt.figure()
        ax_surface_compare = plt.axes()
 
        deltaA = np.flipud(TA.deltaA).transpose()
        scaledTA = np.arctan(deltaA/np.max(deltaA)) # just for plotting! normalize deltaA and use arctan transformation
        deltaA_compare = np.flipud(TA_compare.deltaA).transpose()
        scaledTA_compare = np.arctan(deltaA_compare/np.max(deltaA_compare)) # just for plotting! normalize deltaA and use arctan transformation
        
        # ax_surface_compare.pcolormesh(TA.wavelength[::-1], TA.delay, scaledTA, shading='nearest', cmap=cm.twilight)
        ax_surface_compare.contour(TA.wavelength[::-1], TA.delay, scaledTA, alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10))
        # ax_surface_compare.pcolormesh(TA_compare.wavelength[::-1], TA_compare.delay, scaledTA_compare, shading='nearest', cmap=cm.twilight)
        ax_surface_compare.contour(TA_compare.wavelength[::-1], TA_compare.delay, scaledTA_compare, alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10))
        
        ax_surface_compare.set_xlim((min(TA.wavelength),max(TA.wavelength)))
        
        ax_surface_compare.set_yscale('symlog', linthresh=1.0, linscale=0.35)
        xlims = ax_surface_compare.get_xlim()
        ylims = ax_surface_compare.get_ylim()
        ax_surface_compare.set_xlim(xlims)
        ax_surface_compare.set_ylim(ylims)
        ax_surface_compare.set_xlabel('Wavelength')
        ax_surface_compare.set_ylabel('Delay')
        

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
        self.setup_colors(num_colors=14)
        
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
        self.colors = [to_hex(cm.rainbow(x)) for x in cm_selector]
        self.color_indices = np.arange(num_colors) # the first index (=0) is the always the next color
        
        # associate color tags with row color
        for color in self.colors:
            self.tag_configure(tagname=color, background=color)
            
    def fixed_map(self, option):
        # function I found on the internet that helps to show colored rows in Treeview. If not for this function, the rows will ot be colored
        return [elm for elm in self.style.map("Treeview", query_opt=option)
                if elm[:2] != ("!disabled", "!selected")]