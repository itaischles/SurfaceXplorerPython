import numpy as np
import scipy.linalg
from scipy.special import iv # modified Bessel function of the first kind

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

class Model_2A_B:
    
    def __init__(self):
        
        self.type = 'diffeq'
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

class Model_2A_G_1D_AND_A_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(1D)|A>G'
        self.initial_populations = (1,)
        self.K = (1,10)
        self.lower_bounds_K = (1e-4,0.1)
        self.upper_bounds_K = (1e3,10000)
        self.parameter_names = ('k (biexciton)','tau A->Gnd')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(2*np.sqrt(abs(t))) * y[0]**2 -1/k[1]*y[0]
               ]

#####################################################################################################################
#####################################################################################################################

class Model_A1_G_1D_AND_A2_B:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A1>G(1D)|A2>B'
        self.initial_populations = (0.5,0.5,0)
        self.K = (0.1, 15)
        self.lower_bounds_K = (0.01, 2)
        self.upper_bounds_K = (10, 100)
        self.parameter_names = ('k', 'tau')
    
    def diffeq(self,t,y,k):
        return [
                 -k[0]/(2*np.sqrt(abs(t))) * y[0]**2,
                 -1/k[1]*y[1],
                  1/k[1]*y[1]
                 ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_THEN_2B_1D_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B|2B>G(1D)'
        self.initial_populations = (1,0)
        self.K = (1, 1)
        self.lower_bounds_K = (0.01, 0.01)
        self.upper_bounds_K = (100, 100)
        self.parameter_names = ('tau', 'k')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -k[1]/(2*np.sqrt(abs(t))) * y[1]**2,
                 ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_B_THEN_2B_3D_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B|2B>G(3D)'
        self.initial_populations = (1,0)
        self.K = (1, 1)
        self.lower_bounds_K = (0.01, 0.01)
        self.upper_bounds_K = (100, 100)
        self.parameter_names = ('tau', 'k')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -k[1]*y[1]**2,
                 ]

#####################################################################################################################
#####################################################################################################################

class Model_A_B_THEN_2B_1D_G_AND_B_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>G|2B>G(1D)'
        self.initial_populations = (1,0)
        self.K = (1, 10, 1)
        self.lower_bounds_K = (0.1, 1, 0.01)
        self.upper_bounds_K = (100, 100, 100)
        self.parameter_names = ('tau A->B', 'tau B->Gnd', 'k (biexciton)')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -1/k[1]*y[1] -k[2]/(2*np.sqrt(abs(t))) * y[1]**2,
                 ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_B_C_THEN_2C_1D_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C|2C>G(1D)'
        self.initial_populations = (1,0,0)
        self.K = (1, 10, 1)
        self.lower_bounds_K = (0.1, 0.1, 0.01)
        self.upper_bounds_K = (10, 100, 100)
        self.parameter_names = ('tau A->B', 'tau B->C', 'k')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -1/k[1]*y[1],
                 1/k[1]*y[1] -k[2]/(2*np.sqrt(abs(t))) * y[2]**2,
                 ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_B_C_THEN_2C_3D_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'A>B>C|2C>G(3D)'
        self.initial_populations = (1,0,0)
        self.K = (1,10, 1)
        self.lower_bounds_K = (0.1, 1, 0.01)
        self.upper_bounds_K = (10, 100, 100)
        self.parameter_names = ('tau(A->B)', 'tau(B->C)', 'k(2C->Gnd)')
    
    def diffeq(self,t,y,k):
        return [
                -1/k[0]*y[0],
                 1/k[0]*y[0] -1/k[1]*y[1],
                 1/k[1]*y[1] -k[2]*y[2]**2,
                 ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_G_1D_and_B_G:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(1D)|B>G'
        self.initial_populations = (1,1)
        self.K = (1,1000)
        self.lower_bounds_K = (1e-4,10)
        self.upper_bounds_K = (1e3,5000)
        self.parameter_names = ('k 2A->G','tau B->G')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(2*np.sqrt(abs(t)))*y[0]**2,
                -1/k[1]*y[1]
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_A_G_distribution:
    
    def __init__(self):
        
        self.type = 'other'
        self.name = 'A>G(distribution)'
        self.initial_populations = (1,)
        self.K = (10, 0.1)
        self.lower_bounds_K = (0.1, 0.01)
        self.upper_bounds_K = (10000, 10)
        self.parameter_names = ('tau A->Gnd', 'sigma_k A->Gnd')
        
    def get_species_decay(self,time,params):
        
        # generate distribution of k values
        k = np.logspace(-5,2,100)
        dist_kAG = np.exp( - (np.log10(k)-np.log10(1/params[0]))**2 / (2*params[1])**2 )
        dist_kAG = dist_kAG/max(dist_kAG)
        
        # generate species decay curve from distribution
        inhomo_decay = np.zeros((len(time),1))
        for i,t in enumerate(time):
            if t>0:
                inhomo_decay[i] = np.trapz(dist_kAG*np.exp(-k*t), k)
        
        return inhomo_decay
    
#####################################################################################################################
#####################################################################################################################

class Model_A_B_G_distribution:
    
    def __init__(self):
        
        self.type = 'other'
        self.name = 'A>B>G(distribution)'
        self.initial_populations = (1,0)
        self.K = (10, 0.1, 1000, 0.1)
        self.lower_bounds_K = (0.1, 0.01, 1, 0.01)
        self.upper_bounds_K = (100, 1, 10000, 1)
        self.parameter_names = ('tau A->B', 'sigma_log(tau) A->B', 'tau B->Gnd', 'sigma_log(tau) B->Gnd')
        
    def get_species_decay(self,time,params):
        
        # generate distributions of k values
        k = np.logspace(-5,1,100)
        kX,kY = np.meshgrid(k,k)
        mu = np.array((1/params[0], 1/params[2]))
        sigma = np.array((params[1],params[3]))
        
        # 1-D distribution over k
        dist1 = np.exp(-0.5*((np.log10(k)-np.log10(mu[0]))**2)/np.log10(sigma[0])**2)
        # 2-D distribution over k,k (since 2nd species rises and decays with two time constants/distributions)
        dist2 = np.exp(-0.5*((np.log10(kX)-np.log10(mu[0]))**2)/np.log10(sigma[0])**2 - 0.5*((np.log10(kY)-np.log10(mu[1]))**2)/np.log10(sigma[1])**2)
        dist2 = dist2/np.max(np.max(dist2))
        
        # generate species decay curve from distribution
        inhomo_decay = np.zeros((len(time),2))
        for i,t in enumerate(time):
            if t>0:
                inhomo_decay[i,0] = np.trapz(dist1*np.exp(-k*t), k)
                integrand2 = kX/(kX+1e-10-kY)*dist2*(np.exp(-kY*t)-np.exp(-kX*t))
                inhomo_decay[i,1] = np.trapz(np.trapz(integrand2, k), k)
        
        return inhomo_decay

#####################################################################################################################
#####################################################################################################################