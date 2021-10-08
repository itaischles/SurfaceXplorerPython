import tkinter as tk
from tkinter import ttk

#####################################################################################################################
#####################################################################################################################

class FittingUserInputFrame(tk.Frame):

    def __init__(self, master, fit_model_names, fitmodel=[]):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        # initialize model selection
        self.selected_model = tk.StringVar(self, value=fit_model_names[0])
        self.model_combobox = ttk.Combobox(self)
        self.model_combobox.configure(state='readonly',
                                      values=fit_model_names,
                                      textvariable=self.selected_model,
                                      )
        self.model_label = ttk.Label(self, text='Model:')
        
        # initialize fitting parameter table
        self.fit_params_table = FitParamsTable(self, fitmodel)
        
        # initialize fit button
        self.fit_button = ttk.Button(self, text='Fit', state='disabled')
        
        # use geometry manager to arrange widgets in user input frame
        self.model_label.grid(row=0,column=0,sticky='we')
        self.model_combobox.grid(row=0,column=1,sticky='we',pady=10)
        self.fit_params_table.grid(row=2,column=0,columnspan=2,pady=5)
        self.fit_button.grid(row=3,column=0,columnspan=2,pady=5)
        
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
        
        for i,name in enumerate(fitmodel.model.parameter_names, start=1):
            
            # add labels for fitting parameter names
            self.label.append(tk.Label(self, text=name, justify='left'))
            self.label[i].grid(row=i+1,column=0)
            
            # initialize string variables for the fitting parameters
            self.values.append(tk.StringVar())
            self.values[i].set(str(fitmodel.model.K[i-1]))
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