import tkinter as tk
from tkinter import ttk

from datetime import datetime

#####################################################################################################################
#####################################################################################################################

class FilePathFrame(tk.Frame):
    
    def __init__(self, master):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        # add user tools to the tools frame
        self.filepath_label = ttk.Label(self, text='No file opened. To open a file use File > Open...')
        self.filepath_label.pack()
        
    def set_new_filepath(self, filepath):
        
        self.filepath_label.config(text = filepath)
        

    
#####################################################################################################################
#####################################################################################################################

class LogFrame(tk.Frame):
    
    def __init__(self, master):
        
        super().__init__(master, relief=tk.FLAT) # parent class (=tk.Frame) initialization
        
        # add treeview to display messages and scrollbar to the user info frame
        self.info_log = ttk.Treeview(master=self,
                                     columns=('col#1', 'col#2'),
                                     show='headings',
                                     height=3,
                                     selectmode='none'
                                     )
        self.scrollbar = ttk.Scrollbar(master=self,
                                       orient=tk.VERTICAL,
                                       command=self.info_log.yview
                                       )
        
        self.info_log.heading('col#1', text='Time')
        self.info_log.heading('col#2', text='Info')
        
        self.info_log.column('col#1', width=120)
        self.info_log.column('col#2', width=880)
        
        self.info_log.config(yscrollcommand=self.scrollbar.set)
        
        self.info_log.pack(side=tk.LEFT)
        self.scrollbar.pack(side=tk.LEFT, fill = tk.BOTH)
        
    def update_log(self, new_info):
        now = datetime.now()
        self.info_log.insert('', 0, values=(now.strftime("%Y-%m-%d %H:%M:%S"),new_info))