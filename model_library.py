import numpy as np
import scipy.linalg
from scipy.special import iv # modified Bessel function of the first kind

#####################################################################################################################
############################################ INSTRUCTIONS ###########################################################
#####################################################################################################################
# Instructions on how to create a new model:
#
# 1. You can start by copying an existing model class. Mind the indentation of the code! It is important in Python!
# 2. Change the name of the class (whatever is written after the keyword 'class' and followed by ':') so that it is unique.
# 3. Under the '__init__' method change the model type to 'diffeq' for differential equation type (d[A]/dt=...) or 'direct'
#    if it directly outputs the populations ([A]=...).
# 4. Under the '__init__' method change name to whatever string you would like. This will be the official name of your model.
# 5. In the self.initial_population, make sure you put between parenthesis the correct number of initial species with relative
#    populations. e.g.: (1,0) means species 1 starts at 100% which is then replaced by species 2 which starts at 0%.
#    Important: for vectors containing one element, use a comma: ',' after the last element. This tells Python it is a
#    vector and not a scalar (see e.g. A->Gnd model).
# 6. self.K is a list of model specific parameters (decay rates, time constants,...) excluding the IRF (which is not
#    included in the model itself)
# 7. Add lower bounds and upper bounds on the fitting parameters in the next two vectors. Make sure there is the same number
#    of elements as in the self.K vector.
# 8. Add meaningful parameter names as a list/tuple of strings (in quotes, ', or double-quotes, ")
# 
# FOR 'diffeq' type models:
#    Only the body of the function 'diffeq' should be changed in the following way:
#    After the keyword 'return' there should be an opening square bracket, and in the end a closing square bracket.
#    Between those, add the right-hand-side of the differential equations separated by commas for each equation.
#    k is a vector of the fitting parameter values to be inserted so that k[0] is the first fitting parameter,
#    k[1] is the second, and so on... y is the species population vector so that y[0] is the first species,
#    y[1] is the second species, and so on... To add higher order powers in Python you use double asterisk, e.g.
#    to write 'y[0] squared' use y[0]**2.
# 
# FOR 'direct' type models:
#    Create a method named 'get_species_decay' with input parameters '(self,time,params)' (see e.g. Model_A_G_and_B_G_direct).
#    The output of this method is a [N,M] matrix where N has length of the delay vector, and M has length of the number of
#    species. For example, for a model with 2 species and an output matrix named 'p': p[:,0] is the first population decay
#    curve, and p[:,1] is the second population decay curve.
# 
# If you have questions you can always email me at: itaischles@gmail.com   
#####################################################################################################################
#####################################################################################################################        

class Model_A_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>G'
        self.initial_populations = (1,)
        self.K = (10,)
        self.lower_bounds_K = (0.1,)
        self.upper_bounds_K = (10000,)
        self.parameter_names = ('tau A->Gnd',)
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0]
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_G_stretched:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>G(stretched)'
        self.initial_populations = (1,)
        self.K = (10,0.5)
        self.lower_bounds_K = (0.1,0.1)
        self.upper_bounds_K = (10000,1)
        self.parameter_names = ('tau A->Gnd','beta A->Gnd')
    
    def diffeq(self,t,y,k):
        
        tau = k[0]
        beta = k[1]
        
        if t>0:
            rate = beta/tau**beta * 1/t**(1-beta)
        else:
            rate = 0
        
        return [
                -rate*y[0]
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_B:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B'
        self.initial_populations = (1,0)
        self.K = (10,)
        self.lower_bounds_K = (0.1,)
        self.upper_bounds_K = (10000,)
        self.parameter_names = ('tau A->B',)
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]
               ]
    
#####################################################################################################################
#####################################################################################################################       

class Model_A_B_stretched:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B(stretched)'
        self.initial_populations = (1,0)
        self.K = (10,0.5)
        self.lower_bounds_K = (0.1,0.1)
        self.upper_bounds_K = (10000,1)
        self.parameter_names = ('tau A->B','beta A->B')
    
    def diffeq(self,t,y,k):
        
        tau = k[0]
        beta = k[1]
        
        if t>0:
            rate = beta/tau**beta * 1/t**(1-beta)
        else:
            rate = 0
        
        return [
                -rate*y[0],
                 rate*y[0]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_C:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C'
        self.initial_populations = (1,0,0)
        self.K = (1,100)
        self.lower_bounds_K = (0.1,1)
        self.upper_bounds_K = (10,10000)
        self.parameter_names = ('tau A->B', 'tau B->C')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]-1/k[1]*y[1],
                 1/k[1]*y[1]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_C_D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C>D'
        self.initial_populations = (1,0,0,0)
        self.K = (1,10,100)
        self.lower_bounds_K = (0.1,1,10)
        self.upper_bounds_K = (10,100,10000)
        self.parameter_names = ('tau A->B', 'tau B->C', 'tau C->D')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]-1/k[1]*y[1],
                 1/k[1]*y[1]-1/k[2]*y[2],
                 1/k[2]*y[2]
               ]

#####################################################################################################################
#####################################################################################################################        

class Model_A_B_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>G'
        self.initial_populations = (1,0)
        self.K = (1,100)
        self.lower_bounds_K = (0.1,10)
        self.upper_bounds_K = (100,10000)
        self.parameter_names = ('tau A->B', 'tau B->Gnd')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]-1/k[1]*y[1]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_G_stretched:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>G(stretched)'
        self.initial_populations = (1,0)
        self.K = (1,0.5,100,0.5)
        self.lower_bounds_K = (0.1,0.1,10,0.1)
        self.upper_bounds_K = (100,1,10000,1)
        self.parameter_names = ('tau A->B', 'beta A->B', 'tau B->Gnd', 'beta B->Gnd')
    
    def diffeq(self,t,y,k):
        
        tau1 = k[0]
        beta1 = k[1]
        tau2 = k[2]
        beta2 = k[3]
        
        if t>0:
            rate1 = beta1/tau1**beta1 * 1/t**(1-beta1)
            rate2 = beta2/tau2**beta2 * 1/t**(1-beta2)
        else:
            rate1 = 0
            rate2 = 0
        
        return [
                -rate1*y[0],
                 rate1*y[0]-rate2*y[1]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_C_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C>G'
        self.initial_populations = (1,0,0)
        self.K = (1,100,1000)
        self.lower_bounds_K = (0.1,1,10)
        self.upper_bounds_K = (10,1000,10000)
        self.parameter_names = ('tau A->B', 'tau B->C', 'tau C->Gnd')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]-1/k[1]*y[1],
                 1/k[1]*y[1]-1/k[2]*y[2]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_C_D_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C>D>G'
        self.initial_populations = (1,0,0,0)
        self.K = (1,10,100,1000)
        self.lower_bounds_K = (0.1,1,10,100)
        self.upper_bounds_K = (10,100,1000,10000)
        self.parameter_names = ('tau A->B', 'tau B->C', 'tau C->D', 'tau D->Gnd')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0]-1/k[1]*y[1],
                 1/k[1]*y[1]-1/k[2]*y[2],
                 1/k[2]*y[2]-1/k[3]*y[3]
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_G_3D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(3D)'
        self.initial_populations = (1,)
        self.K = (1,)
        self.lower_bounds_K = (1e-4,)
        self.upper_bounds_K = (1e3,)
        self.parameter_names = ('k 2A->G',)
    
    def diffeq(self,t,y,k):
        return [
                -k[0]*y[0]**2
               ]

#####################################################################################################################
#####################################################################################################################

class Model_2A_G_2D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(2D)'
        self.initial_populations = (1,)
        self.K = (1,)
        self.lower_bounds_K = (1e-4,)
        self.upper_bounds_K = (1e3,)
        self.parameter_names = ('k 2A->G',)
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(np.log(abs(t)+np.exp(1/2))) * y[0]**2
               ]

#####################################################################################################################
#####################################################################################################################

class Model_2A_G_1D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(1D)'
        self.initial_populations = (1,)
        self.K = (1,)
        self.lower_bounds_K = (1e-4,)
        self.upper_bounds_K = (1e3,)
        self.parameter_names = ('k 2A->G',)
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(2*np.sqrt(abs(t)+0.25)) * y[0]**2
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_B_3D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>B(3D)'
        self.initial_populations = (1,0)
        self.K = (1,)
        self.lower_bounds_K = (1e-4,)
        self.upper_bounds_K = (1e3,)
        self.parameter_names = ('k 2A->B',)
    
    def diffeq(self,t,y,k):
        return [
                -k[0]*y[0]**2,
                 k[0]*y[0]**2
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_B_1D:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>B(1D)'
        self.initial_populations = (1,0)
        self.K = (1,)
        self.lower_bounds_K = (1e-4,)
        self.upper_bounds_K = (1e3,)
        self.parameter_names = ('k 2A->B',)
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(2*np.sqrt(abs(t)+0.25))*y[0]**2,
                 k[0]/(2*np.sqrt(abs(t)+0.25))*y[0]**2
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_B_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>B>G'
        self.initial_populations = (1,0)
        self.K = (1,10)
        self.lower_bounds_K = (1e-4,0.1)
        self.upper_bounds_K = (1e3,10000)
        self.parameter_names = ('k 2A->B','tau B->Gnd')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]*y[0]**2,
                 k[0]*y[0]**2-1/k[1]*y[1]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A_G_and_B_G_direct:
    
    def __init__(self):
        
        self.type = 'direct'
        self.name = 'A>G|B>G'
        self.initial_populations = (1,1)
        self.K = (10,1000)
        self.lower_bounds_K = (0.1,10)
        self.upper_bounds_K = (1000,10000)
        self.parameter_names = ('tau A->Gnd','tau B->Gnd')
        
    def get_species_decay(self, time, params):
        
        # extract parameters from input and give them meaningful names
        tauA = params[0]
        tauB = params[1]
        
        # initialize decay matrix to zeros
        decays = np.zeros((len(time),2))

        # populate decay curves for populations (only where time>0)
        decays[time>0, 0] = np.exp(-time[time>0]/tauA)
        decays[time>0, 1] = np.exp(-time[time>0]/tauB)
        
        # return the decay curves
        return decays