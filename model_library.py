import numpy as np

#####################################################################################################################
############################################ INSTRUCTIONS ###########################################################
#####################################################################################################################
# Instructions on how to create a new model:
# 1. You can start by copying an existing model class. Mind the indentation of the code! It is important in Python!
# 2. Change the name of the class (whatever is written after the keyword 'class' and followed by ':') so that it is unique.
# 3. Under the '__init__' method change name to whatever string you would like. This will be the official name of your model.
# 4. In the self.initial_population, make sure you put between parenthesis the correct number of initial species with relative
#    populations. e.g.: (1,0) means species 1 starts at 100% which is then replaced by species 2 which starts at 0%.
#    Important: for vectors containing one element, use a comma: ',' after the last element. This tells Python it is a
#    vector and not a scalar.
# 5. self.K is a list/tuple of model specific parameters (decay rates, time constants,...) excluding the IRF (which is not
#    included in the model itself)
# 6. Add lower bounds and upper bounds on the fitting parameters in the next two vectors. Make sure there is the same number
#    of elements as in the self.K vector.
# 7. Add parameters names as a list/tuple of strings (in quotes)
# 8. Only the body of the function 'diffeq' should be changed in the following way:
#    After the keyword 'return' there should be an opening square bracket and in the end closing square bracket.
#    Between those, add the right-hand-side of the differential equations separated by commas for each equation.
#    k is a vector of the fitting parameter values to be inserted so that k[0] is the first fitting parameter,
#    k[1] is the second, and so on... y is the species population vector so that y[0] is the first species,
#    y[1] is the second species, and so on... To add higher order powers in Python you use double asterisk, e.g.
#    to write 'y[0] squared' use y[0]**2.
# 9. If you have questions you can always email me at: itaischles@gmail.com
#####################################################################################################################
#####################################################################################################################        

class Model_A_G:
    
    def __init__(self):
        
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

class Model_A_B:
    
    def __init__(self):
        
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

class Model_A_B_C:
    
    def __init__(self):
        
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

class Model_A_B_C_G:
    
    def __init__(self):
        
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

class Model_2A_B:
    
    def __init__(self):
        
        self.name = '2A>B'
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

class Model_2A_B_G:
    
    def __init__(self):
        
        self.name = '2A>B>G'
        self.initial_populations = (1,0)
        self.K = (1,10)
        self.lower_bounds_K = (1e-4,0.1)
        self.upper_bounds_K = (1e3,10000)
        self.parameter_names = ('k 2A->B','tau A->B')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]*y[0]**2,
                 k[0]*y[0]**2-1/k[1]*y[1]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_B1_G_biexciton_AND_B2_C:
    
    def __init__(self):
        
        self.name = 'A1>G(biexciton)|A2>B'
        self.initial_populations = (0.5,0.5,0)
        self.K = (0.1, 15)
        self.lower_bounds_K = (0.01, 2)
        self.upper_bounds_K = (10, 100)
        self.parameter_names = ('k', 'tau')
    
    def diffeq(self,t,y,k):
        return [
                 -k[0]/(2*np.sqrt(abs(t)))*y[0]**2,
                 -1/k[1]*y[1],
                  1/k[1]*y[1]
                 ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_THEN_2B_G:
    
    def __init__(self):
        
        self.name = 'A>B|2B>G'
        self.initial_populations = (1,0)
        self.K = (1, 1)
        self.lower_bounds_K = (0.01, 0.01)
        self.upper_bounds_K = (100, 100)
        self.parameter_names = ('tau', 'k')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -k[1]*y[1]**2
                 ]