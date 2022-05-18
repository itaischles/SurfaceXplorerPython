import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm # color maps
from matplotlib.colors import to_hex, to_rgb
import matplotlib.colors as colors

class PlotAreaFrame(tk.Frame):
    
    def __init__(self, master):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        # initialize right side frame
        self.frame_right = tk.Frame(self)
        
        # create plot toolbar frame
        self.toolbar_frame = tk.Frame(self)
        
        # create the figure
        self.fig = plt.Figure(figsize=(8,6),dpi=100)
        self.fig.set_tight_layout(True)
        
        # create axes
        self.ax1 = self.fig.add_subplot(2,2,1)
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)
        
        self.residuals_colorbar = self.fig.colorbar(self.ax4.pcolormesh((0,1),(0,1),((0,0),(0,0)),shading='auto',cmap=cm.RdBu_r), ax=self.ax4, label='%')
        
        # make Tk/Matplotlib figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        
        # create toolbars
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack()
    
        # create tree widget to hold POIs
        self.POI_table = POITable(self.frame_right)
        
        # create scrollbar for POI table
        self.POI_table_scrollbar = ttk.Scrollbar(master=self.frame_right,
                                                 orient='vertical',
                                                 command=self.POI_table.yview
                                                 )
        self.POI_table.config(yscrollcommand=self.POI_table_scrollbar.set)
        
        # create header label for POI table
        self.POI_label = ttk.Label(self.frame_right,text='Points of interest')
        
        # create header for display types
        self.display_label = tk.Label(self.frame_right, text='Display type selection:')
        
        # create radio buttons for selecting what to display on the figures
        self.display_var = tk.IntVar()
        self.display_POIs_radiobutton = ttk.Radiobutton(self.frame_right, text='Points of interest', variable=self.display_var, value=0, state='disabled')
        self.display_globalfitting_radiobutton = ttk.Radiobutton(self.frame_right, text='Global fitting analysis', variable=self.display_var, value=1, state='disabled')
        self.display_mcrals_radiobutton = ttk.Radiobutton(self.frame_right, text='MCR-ALS analysis', variable=self.display_var, value=2, state='disabled')
        
        # use geometry managers to pack the widgets
        self.POI_label.grid(row=0,column=0,sticky='n')
        self.POI_table.grid(row=1,column=0,sticky='ns')
        self.POI_table_scrollbar.grid(row=1,column=1,sticky='ns')
        self.display_label.grid(row=2,column=0,sticky='w')
        self.display_POIs_radiobutton.grid(row=3,column=0,sticky='w')
        self.display_globalfitting_radiobutton.grid(row=4,column=0,sticky='w')
        self.display_mcrals_radiobutton.grid(row=5,column=0,sticky='w')
        
        # Use the grid geometry manager to pack the widgets in the main plot area frame
        self.canvas.get_tk_widget().grid(row=0,column=0,sticky='nsew')
        self.frame_right.grid(row=0,column=1,sticky='ns')
        self.toolbar_frame.grid(row=1,column=0,sticky='we')
        
        # resizing
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
    def refresh_fig1(self, TA, fitmodel=[]):
        
        # start by plotting the TA surface
        if TA == []:
            return
        
        sparse_step = 5 # for plotting speed, plot every 'sparse_step' pixels of data
        
        self.ax1.cla()

        deltaA = np.flipud(TA.deltaA).transpose()
        scaledTA = np.tanh(deltaA) # just for plotting! normalize deltaA and use tanh transformation
        self.ax1.contourf(TA.wavelength[::-sparse_step], TA.delay, scaledTA[::1,::sparse_step], cmap=cm.RdBu_r, levels=40, norm=colors.CenteredNorm())
        self.ax1.contour(TA.wavelength[::-sparse_step], TA.delay, scaledTA[::1,::sparse_step], colors='black', levels=40, alpha=0.3, linewidths=0.2)
        
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
        
        # get best fit parameters from model
        fit_params = np.concatenate(([fitmodel.irf, fitmodel.tzero], fitmodel.model.K))
        
        if self.display_var.get() == 0:
            
            # plot the experimental and fitted kinetic traces overlaid
            for i,iid in enumerate(self.POI_table.get_children()):
                wavelength, delay, color = self.POI_table.get_poi(iid)
                wavelength_index = np.absolute(wavelength-TA.wavelength).argmin()
                self.ax2.plot(TA.delay, TA.deltaA[wavelength_index,:]*1000, color=color, marker='o', linestyle='none', markersize=3, alpha=0.25, markeredgecolor='none')
                self.ax2.plot(TA.delay, fitmodel.calc_model_deltaA(fit_params)[wavelength_index,:]*1000, color=color, lw=1)
            
            # update axis labels
            self.ax2.set_ylabel(r'$\Delta$A $\times 10^3$')
            
        elif self.display_var.get() == 1:
            
            # plot the fitted populations from the global analysis
            model_populations = fitmodel.calc_species_decays(fit_params)
            self.ax2.plot(TA.delay, model_populations, lw=1)
            
            # update axis labels
            self.ax2.set_ylabel('Norm. populations')
            
        elif self.display_var.get() == 2:
            
            # plot the fitted populations from the MCR-ALS analysis
            if fitmodel.mcrals.n_targets is None:
                tk.messagebox.showerror('No MCR-ALS fit found', 'Please provide MCR-ALS fit first using "Fitting > Perform MCR-ALS analysis..."')
                self.display_var.set(0) # go back to POI view-type
                self.refresh_fig2(TA, fitmodel)
                return
            
            mcrals_populations = fitmodel.mcrals.C_opt_
            model_populations = fitmodel.calc_species_decays(fit_params)
            
            for i in range(model_populations.shape[1]):
                color = cm.tab10(i)
                self.ax2.scatter(TA.delay, mcrals_populations[:,i], s=1, color=color, alpha=0.5)
                self.ax2.plot(TA.delay, model_populations[:,i], lw=1, color=color)
            
            self.ax2.set_xlim((min(TA.delay), max(TA.delay)))
            
            # update axis labels
            self.ax2.set_ylabel('Norm. populations')
            
        else:
            
            return
            
            model_populations = fitmodel.mcrals.C_opt_
            self.ax2.plot(TA.delay, model_populations, lw=1)
            
            # update axis labels
            self.ax2.set_ylabel('Norm. populations')
            
        self.ax2.set_xlabel('Delay')
        self.ax2.set_xscale('symlog', linthresh=1.0, linscale=0.35)
        self.ax2.set_xlim((min(TA.delay),max(TA.delay)))
                
        # add line at deltaA=0
        self.ax2.plot([min(TA.delay), max(TA.delay)],[0,0], linewidth=0.3, color=(0.2,0.2,0.2))
        
        # show plot on GUI
        self.ax2.figure.canvas.draw()
        
    def refresh_fig3(self, TA, fitmodel=[]):
        
        if TA == []:
            return
        
        self.ax3.cla()
        
        # get best fit parameters from model
        fit_params = np.concatenate(([fitmodel.irf, fitmodel.tzero], fitmodel.model.K))
        
        if self.display_var.get() == 0:
            
            # plot the spectra of the selected wavelengths (in POIs)
            for iid in self.POI_table.get_children():
                wavelength, delay, color = self.POI_table.get_poi(iid)
                delay_index = np.argmin(abs(TA.delay-delay))
                self.ax3.plot(TA.wavelength, TA.deltaA[:,delay_index]*1000, lw=1, color=color, alpha=0.75)
                self.ax3.set_ylabel(r'$\Delta$A $\times 10^3$')
        
        elif self.display_var.get() == 1:
            
            # plot the fitted species spectra from the global analysis
            species_spectra = fitmodel.normalize_species_spectra(fitmodel.calc_model_species_spectra(fit_params))
            self.ax3.plot(TA.wavelength, species_spectra, lw=1)
            self.ax3.set_ylabel('Norm. species spectra')
            
        elif self.display_var.get() == 2:
            
            # plot the fitted species spectra from the MCR-ALS analysis
            if fitmodel.mcrals.n_targets is None:
                tk.messagebox.showerror('No MCR-ALS fit found', 'Please provide MCR-ALS fit first using "Fitting > Perform MCR-ALS analysis..."')
                self.display_var.set(0)
                self.refresh_fig3(TA, fitmodel)
                return
            
            mcrals_spectra = fitmodel.normalize_species_spectra(fitmodel.mcrals.ST_opt_.T)
            self.ax3.plot(TA.wavelength, mcrals_spectra, lw=1)
            
            # update axis labels
            self.ax3.set_ylabel('Norm. species spectra')
            
        else:
            
            return
        
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
        
        sparse_step = 5 # for plotting speed, plot every 'sparse_step' pixels of data
        
        self.ax4.cla()
        
        if fitmodel != []:
            
            deltaA_residuals = fitmodel.residuals_matrix
            deltaA_rel_residuals = np.clip(deltaA_residuals/(TA.deltaA+1e-9)*100, -100, 100)
            self.residuals_plot = self.ax4.contourf(TA.wavelength[::sparse_step], TA.delay, deltaA_rel_residuals.T[::1,::sparse_step], cmap=cm.RdBu_r, norm=colors.CenteredNorm(), levels=40)
            self.residuals_colorbar.update_normal(self.residuals_plot)
            
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
        scaledTA = np.tanh(deltaA) # just for plotting! normalize deltaA and use arctan transformation
        deltaA_compare = np.flipud(TA_compare.deltaA).transpose()
        scaledTA_compare = np.arctan(deltaA_compare/np.max(deltaA_compare)) # just for plotting! normalize deltaA and use arctan transformation
        
        # ax_surface_compare.pcolormesh(TA.wavelength[::-1], TA.delay, scaledTA, shading='nearest', cmap=cm.twilight)
        ax_surface_compare.contour(TA.wavelength[::-1], TA.delay, scaledTA, alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10), colors='red')
        # ax_surface_compare.pcolormesh(TA_compare.wavelength[::-1], TA_compare.delay, scaledTA_compare, shading='nearest', cmap=cm.twilight)
        ax_surface_compare.contour(TA_compare.wavelength[::-1], TA_compare.delay, scaledTA_compare, alpha=0.3, linewidths=0.5, linestyles='solid', levels=np.linspace(-1., 1., 10), colors='blue')
        
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