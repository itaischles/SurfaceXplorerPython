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
# 3. Under the '__init__' method change the model type to 'diffeq' for differential equation type (d[A]/dt=...) or 'other'
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
# FOR 'other' type models:
#    Create a method named 'get_species_decay' with input parameters '(self,time,params)' (see e.g. Model_A_G_distribution).
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

class Model_2A_G_3D_and_invisible_B:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(3D)|?>B'
        self.initial_populations = (1,0)
        self.K = (1,100)
        self.lower_bounds_K = (1e-4,1e-1)
        self.upper_bounds_K = (1e3,1e4)
        self.parameter_names = ('k 2A->B','tau ?>B')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]*y[0]**2,
                 1/k[1]*(1-y[1])
               ]
    
#####################################################################################################################
#####################################################################################################################

class Model_2A_G_1D_and_invisible_B:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = '2A>G(1D)|?>B'
        self.initial_populations = (1,0)
        self.K = (1,100)
        self.lower_bounds_K = (1e-4,1e-1)
        self.upper_bounds_K = (1e3,1e4)
        self.parameter_names = ('k 2A->B','tau ?>B')
    
    def diffeq(self,t,y,k):
        return [
                -k[0]/(2*np.sqrt(abs(t)+0.25))*y[0]**2,
                 1/k[1]*(1-y[1])
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
                -k[0]/(2*np.sqrt(abs(t)+0.25)) * y[0]**2 -1/k[1]*y[0]
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
                 -k[0]/(2*np.sqrt(abs(t)+0.25)) * y[0]**2,
                 -1/k[1]*y[1],
                  2/k[1]*y[1]
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
                 1/k[0]*y[0] -k[1]/(2*np.sqrt(abs(t)+0.25)) * y[1]**2,
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
                 1/k[0]*y[0] -1/k[1]*y[1] -k[2]/(2*np.sqrt(abs(t)+0.25)) * y[1]**2,
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
                 1/k[1]*y[1] -k[2]/(2*np.sqrt(abs(t)+0.25)) * y[2]**2,
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
                -k[0]/(2*np.sqrt(abs(t)+0.25))*y[0]**2,
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

class Model_WSe2_bare:
    
    def __init__(self):
        
        self.type = 'other'
        self.name = 'WSe2_bare'
        self.initial_populations = (1,0)
        self.K = (10,10,1000)
        self.lower_bounds_K = (0.1,0.1,0.1)
        self.upper_bounds_K = (10000,10000,10000)
        self.parameter_names = ('tau A->Gnd','tau ?->B','tau B->Gnd')
    
    def get_species_decay(self,time,params):
        
        decays = np.zeros((len(time),2))
        decays[time>0,0] = np.exp(-time[time>0]/params[0])
        decays[time>0,1] = (1-np.exp(-time[time>0]/params[1])) * np.exp(-time[time>0]/params[2])
        return decays

#####################################################################################################################
#####################################################################################################################

class Model_WSe2_CoPc:
    
    def __init__(self):
        
        self.type = 'other'
        self.name = 'WSe2_CoPc'
        self.initial_populations = (1,0)
        self.K = (1,0.5,3,500,0.5)
        self.lower_bounds_K = (0.1,0.1,0.1,0.1,0.1)
        self.upper_bounds_K = (10000,1,10000,10000,1)
        self.parameter_names = ('tau A->Gnd','beta A->Gnd','tau ?->B','tau B->Gnd','beta B->Gnd')
    
    def get_species_decay(self,time,params):
        
        decays = np.zeros((len(time),2))
        decays[:,0] = self.stretched_decay(time,params[0],params[1])
        decays[:,1] = self.rise(time,params[2]) * self.stretched_decay(time,params[3],params[4])
        return decays
    
    def stretched_decay(self,t,t0,beta):
        y = np.zeros(len(t))
        y[t>0] = np.exp(-(t[t>0]/t0)**beta)
        return y
    
    def rise(self,t,t0):
        y = np.zeros(len(t))
        y[t>0] = 1-np.exp(-t[t>0]/t0)
        return y

#####################################################################################################################
#####################################################################################################################

class Model_CoroneneNDA:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'Cor-NDA'
        self.initial_populations = (1,0)
        self.K = (1,0.5,100,0.5)
        self.lower_bounds_K = (0.1,0.1,0.1,0.1)
        self.upper_bounds_K = (10000,1,10000,1)
        self.parameter_names = ('tau A->Gnd','beta A->Gnd','tau ?->B','beta ?->B')
    
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
                 rate2*(1-y[1])
               ]
    
#####################################################################################################################
#####################################################################################################################
 
class Model_CoronenePDI:
    
    def __init__(self):
        
        self.type = 'diffeq'
        self.name = 'Cor-PDI'
        self.initial_populations = (1,0,0)
        self.K = (1,100,0.5)
        self.lower_bounds_K = (0.1,0.1,0.1)
        self.upper_bounds_K = (10000,10000,1)
        self.parameter_names = ('tau A->B','tau B->C','beta B->C')
    
    def diffeq(self,t,y,k):
        
        tau1 = k[0]
        tau2 = k[1]
        beta2 = k[2]
        
        if t>0:
            rate1 = 1/tau1
            rate2 = beta2/tau2**beta2 * 1/t**(1-beta2)
        else:
            rate1 = 0
            rate2 = 0
        
        return [
                -rate1*y[0],
                 rate1*y[0]-rate2*y[1],
                 rate2*y[1]
               ]