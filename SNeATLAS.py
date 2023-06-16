import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import nuflux

# integration function changing variable to log, i.e. int f(x) dx --> integrate in u=logx
# x is logspace, y is the function
def integrate_logspace(x, y, a, b):
    interp = sp.interpolate.interp1d(np.log10(x), y, kind='quadratic', fill_value="extrapolate")
    def integrand(log10x):
        return interp(log10x)* 10**(log10x) * np.log(10)
    integral = sp.integrate.quad(integrand, np.log10(a), np.log10(b), limit=100000)
    return integral

########################################################################################################################
# 
########################################################################################################################
class make_NeutrinoEnsemble(): 
    
    def __init__(self, initialFlavorRatio): 
        self.initialFlavorRatio = initialFlavorRatio
        self.PMNSmatrix = self.get_PMNSmatrix()

    theta_12 = 0.563942
    theta_13 = 0.154085
    theta_23 = 0.785398
    delta_m2_12 = 7.65e-05 #units of eV2
    delta_m2_13 = delta_m2_23 = 0.00247 #units of eV2
    deltaCP = 0.0
    hbar = 6.582e-16 #units of eV.s
    c_light = 2.998e8 #units of m/s
    
    def get_PMNSmatrix(self):
        
        theta_12 = self.theta_12; theta_13 = self.theta_13; theta_23 = self.theta_23
    
        s12 = np.sin(theta_12); s13 = np.sin(theta_13); s23 = np.sin(theta_23)
        c12 = np.cos(theta_12); c13 = np.cos(theta_13); c23 = np.cos(theta_23)
        matrix = np.zeros((3,3))

        matrix[0,0] = c12*c13
        matrix[0,1] = s12*c13
        matrix[0,2] = s13

        matrix[1,0] = -s12*c23 - c12*s23*s13
        matrix[1,1] = c12*c23 - s12*s13*s23
        matrix[1,2] = c13*s23

        matrix[2,0] = s12*s23 - c12*s13*c23
        matrix[2,1] = -c12*s23 - s12*s13*c23
        matrix[2,2] = c13*c23

        return matrix
    
    def get_L_E(self, L, E): 
        L = L/(self.hbar * self.c_light)
        return L/E
    
    # large L/E 
    def get_largeLEOscProb(self, L, E, a, b): 
        
        delta_m2_12 = self.delta_m2_12; delta_m2_13 = self.delta_m2_13; delta_m2_23 = self.delta_m2_23
    
        matrix = self.PMNSmatrix

        t1 = matrix[a,0]*matrix[b,0]*matrix[a,1]*matrix[b,1] * np.sin(1.27 * delta_m2_12 * self.get_L_E(L,E))**2
        t2 = matrix[a,0]*matrix[b,0]*matrix[a,2]*matrix[b,2]
        t3 = matrix[a,1]*matrix[b,1]*matrix[a,2]*matrix[b,2]

        if a != b: 
            return -4*(t1 + 0.5*t2 + 0.5*t3)
        elif a == b:
            return 1-4*(t1 + 0.5*t2 + 0.5*t3)
        
    def get_largeLEFinalFlavorRatio(self, L, E):
        #relative to initial flavor ratio     
        nu_e_ratio = self.get_largeLEOscProb(L,E,0,0)*self.initialFlavorRatio[0]\
                     +self.get_largeLEOscProb(L,E,1,0)*self.initialFlavorRatio[1]\
                     +self.get_largeLEOscProb(L,E,2,0)*self.initialFlavorRatio[2]
        nu_mu_ratio = self.get_largeLEOscProb(L,E,0,1)*self.initialFlavorRatio[0]\
                      +self.get_largeLEOscProb(L,E,1,1)*self.initialFlavorRatio[1]\
                      +self.get_largeLEOscProb(L,E,2,1)*self.initialFlavorRatio[2]
        nu_tau_ratio = self.get_largeLEOscProb(L,E,0,2)*self.initialFlavorRatio[0]\
                       +self.get_largeLEOscProb(L,E,1,2)*self.initialFlavorRatio[1]\
                       +self.get_largeLEOscProb(L,E,2,2)*self.initialFlavorRatio[2]
        return [nu_e_ratio,nu_mu_ratio,nu_tau_ratio] / (nu_e_ratio+nu_mu_ratio+nu_tau_ratio)
    
    
    def get_OscProb(self, L, E, a, b): #assumes vanishing CP phase
        
        delta_m2_12 = self.delta_m2_12; delta_m2_13 = self.delta_m2_13; delta_m2_23 = self.delta_m2_23
    
        matrix = self.PMNSmatrix

        t1 = matrix[a,0]*matrix[b,0]*matrix[a,1]*matrix[b,1] * np.sin(1.27 * delta_m2_12 * self.get_L_E(L,E))**2
        t2 = matrix[a,0]*matrix[b,0]*matrix[a,2]*matrix[b,2] * np.sin(1.27 * delta_m2_13 * self.get_L_E(L,E))**2
        t3 = matrix[a,1]*matrix[b,1]*matrix[a,2]*matrix[b,2] * np.sin(1.27 * delta_m2_23 * self.get_L_E(L,E))**2

        if a != b: 
            return -4*(t1 + t2 + t3)
        elif a == b:
            return 1-4*(t1 + t2 + t3)
        
    def get_FinalFlavorRatio(self, L, E):
        #relative to initial flavor ratio     
        nu_e_ratio = self.get_OscProb(L,E,0,0)*self.initialFlavorRatio[0]\
                     +self.get_OscProb(L,E,1,0)*self.initialFlavorRatio[1]\
                     +self.get_OscProb(L,E,2,0)*self.initialFlavorRatio[2]
        nu_mu_ratio = self.get_OscProb(L,E,0,1)*self.initialFlavorRatio[0]\
                      +self.get_OscProb(L,E,1,1)*self.initialFlavorRatio[1]\
                      +self.get_OscProb(L,E,2,1)*self.initialFlavorRatio[2]
        nu_tau_ratio = self.get_OscProb(L,E,0,2)*self.initialFlavorRatio[0]\
                       +self.get_OscProb(L,E,1,2)*self.initialFlavorRatio[1]\
                       +self.get_OscProb(L,E,2,2)*self.initialFlavorRatio[2]
        return [nu_e_ratio,nu_mu_ratio,nu_tau_ratio] / (nu_e_ratio+nu_mu_ratio+nu_tau_ratio)


    
########################################################################################################################
# 
########################################################################################################################
class make_SNeNeutrinoSpectrumFromMurase():
    #'''fluxes from here https://github.com/pegasuskmurase/ModelTemplates'''
    # units originally in eV, s, converted to GeV, s during integration 
    
    def __init__(self, filename):
        self.SNeFluxDF = pd.read_csv(filename, header=None, names=self.col_names, sep=' ')
        
        self.block_num = int( self.SNeFluxDF.shape[0]/(self.block_size) )
        
        self.SNeFluxDF['k'] = np.repeat(np.arange(self.block_num) , self.block_size) #make a column labelling time block k
        
        if ('II-PD1_0Rw1e15' in filename) or ('/IIn/' in filename):
            self.SNeFluxDF['time'] = 1e5 * 10**(0.1*np.repeat(np.arange(self.block_num) , self.block_size))
        else:
            self.SNeFluxDF['time'] = 1e3 * 10**(0.1*np.repeat(np.arange(self.block_num) , self.block_size)) #actual time in s
        
        self.SNeFluxDF['dLumi_neutrino_dE'] = self.SNeFluxDF['dN_neutrino_dE']\
                                             *self.SNeFluxDF['CR_isallowed']\
                                             *self.SNeFluxDF['Energy']*self.SNeFluxDF['Energy']
        
        #after this, convert units to GeV
        self.Energy_array = np.asarray(self.SNeFluxDF['Energy']).reshape((self.block_num, self.block_size)) / 1e9
        self.time_array = np.asarray(self.SNeFluxDF['time']).reshape((self.block_num, self.block_size))
        
        self.dN_neutrino_dE = np.asarray( self.SNeFluxDF['dN_neutrino_dE'] * self.SNeFluxDF['CR_isallowed'] ) * 1e9
        self.dLumi_neutrino_dE = np.asarray(self.SNeFluxDF['dLumi_neutrino_dE']) / 1e9
    
    block_size = 384 #each time point has a spectrum in energy of 384 data points
    
    #column names for the dataframe based on Murase description in the repository
    col_names =  ['Energy', 
                  'IGNORE2',
                  'dN_gamma_dE_noEBL',
                  'dN_gamma_dE_beforeattenuation',
                  'IGNORE5',
                  'IGNORE6',
                  'dN_gamma_dE_generatedgammas',
                  'dN_neutrino_dE',
                  'dN_gamma_dE_optical',
                  'dN_gamma_dE_thermalbrem',
                  'IGNORE11',
                  'IGNORE12',
                  'IGNORE13',
                  'CR_isallowed']
    
    # helpful function to get the time range in which CR_isallowed==1
    def get_CRtimeWindow(self):
        nonzeroCR = self.SNeFluxDF[self.SNeFluxDF['CR_isallowed']==1]
        starttime = np.asarray(nonzeroCR['time'])[0]
        endtime = np.asarray(nonzeroCR['time'])[-1]
        return [starttime, endtime]
    
    # time integrate the neutrino number at each energy slice
    # ======== # ======== # ======== # ======== # ======== # ======== # ======== # ======== # 
    def get_dN_neutrino_dE_atEnergy(self, time, energy_index): 
        time_axis = self.time_array[:,0]
        dN_neutrino_dE = self.dN_neutrino_dE.reshape((self.block_num, self.block_size))
        dNdE = dN_neutrino_dE[:,energy_index]
        return np.interp(time, time_axis, dNdE)
    
    def get_timeIntegrated_dN_neutrino_dE(self):
        
        timeIntegrated_dN_neutrino_dE = np.zeros(self.block_size)
        time_axis = self.time_array[:,0]
        dN_neutrino_dE = self.dN_neutrino_dE.reshape((self.block_num, self.block_size))
        
        for energy_index in range(self.block_size):
            # timeIntegrated_dN_neutrino_dE[energy_index]\
            #  = sp.integrate.quad(self.get_dN_neutrino_dE_atEnergy, time_axis[0], time_axis[-1], args=(energy_index,),limit=10000)[0]
            
            dNdE = dN_neutrino_dE[:,energy_index]
            timeIntegrated_dN_neutrino_dE[energy_index] = integrate_logspace(time_axis, dNdE, time_axis[0], time_axis[-1])[0]
    
        return timeIntegrated_dN_neutrino_dE
    
    def get_timeIntegrated_dN_neutrino_dE_timerange(self, time_intrange):#time_intrange in seconds
        
        timeIntegrated_dN_neutrino_dE = np.zeros(self.block_size)
        time_axis = self.time_array[:,0]
        dN_neutrino_dE = self.dN_neutrino_dE.reshape((self.block_num, self.block_size))
        
        for energy_index in range(self.block_size):
            # timeIntegrated_dN_neutrino_dE[energy_index]\
            #  = sp.integrate.quad(self.get_dN_neutrino_dE_atEnergy, time_axis[0], time_axis[-1], args=(energy_index,),limit=10000)[0]
            
            dNdE = dN_neutrino_dE[:,energy_index]
            timeIntegrated_dN_neutrino_dE[energy_index] = integrate_logspace(time_axis, dNdE, time_axis[0], time_axis[0]+time_intrange)[0]
    
        return timeIntegrated_dN_neutrino_dE
    
    # time integrate the neutrino luminosity at each energy slice
    # ======== # ======== # ======== # ======== # ======== # ======== # ======== # ======== # 
    def get_dLumi_neutrino_dE_atEnergy(self, time, energy_index): 
        time_axis = self.time_array[:,0]
        
        dLumi_neutrino_dE = self.dLumi_neutrino_dE.reshape((self.block_num, self.block_size))
        dLumidE = dLumi_neutrino_dE[:,energy_index]
        
        return np.interp(time, time_axis, dLumidE)
    
    def get_timeIntegrated_dLumi_neutrino_dE(self):
        
        timeIntegrated_dLumi_neutrino_dE = np.zeros(self.block_size)
        time_axis = self.time_array[:,0]
        
        for energy_index in range(self.block_size):
            timeIntegrated_dLumi_neutrino_dE[energy_index]\
             = sp.integrate.quad(self.get_dLumi_neutrino_dE_atEnergy, time_axis[0], time_axis[-1], args=(energy_index,),limit=10000)[0]
    
        return timeIntegrated_dLumi_neutrino_dE
    
    def get_timeIntegrated_dLumi_neutrino_dE_timerange(self, time_intrange):
        
        timeIntegrated_dLumi_neutrino_dE = np.zeros(self.block_size)
        time_axis = self.time_array[:,0]
        
        for energy_index in range(self.block_size):
            timeIntegrated_dLumi_neutrino_dE[energy_index]\
             = sp.integrate.quad(self.get_dLumi_neutrino_dE_atEnergy, time_axis[0], time_axis[0]+time_intrange, args=(energy_index,),limit=10000)[0]
    
        return timeIntegrated_dLumi_neutrino_dE
    

########################################################################################################################
# 
########################################################################################################################
class make_crossSections:
    #units in GeV, cm^2
    
    def __init__(self): 
        self.cs_nu_mu_cc_n = np.loadtxt("nu_cross_sections/nu_mu_H2_cc_n.txt", dtype=float)
        self.cs_nu_mu_cc_p = np.loadtxt("nu_cross_sections/nu_mu_H2_cc_p.txt", dtype=float)
        self.cs_nu_mu_bar_cc_n = np.loadtxt("nu_cross_sections/nu_mu_bar_H2_cc_n.txt", dtype=float)
        self.cs_nu_mu_bar_cc_p = np.loadtxt("nu_cross_sections/nu_mu_bar_H2_cc_p.txt", dtype=float)
        
    def get_cs_nu_mu_cc_n(self, energy):
        return np.interp(energy, self.cs_nu_mu_cc_n[:,0], self.cs_nu_mu_cc_n[:,1]*1e-38)
    def get_cs_nu_mu_cc_p(self, energy):
        return np.interp(energy, self.cs_nu_mu_cc_p[:,0], self.cs_nu_mu_cc_p[:,1]*1e-38)
    def get_cs_nu_mu_bar_cc_n(self, energy):
        return np.interp(energy, self.cs_nu_mu_bar_cc_n[:,0], self.cs_nu_mu_bar_cc_n[:,1]*1e-38)
    def get_cs_nu_mu_bar_cc_p(self, energy):
        return np.interp(energy, self.cs_nu_mu_bar_cc_p[:,0], self.cs_nu_mu_bar_cc_p[:,1]*1e-38)
    
    def get_cs_nu_mu_cc_iso(self, energy):
        cs_p = np.interp(energy, self.cs_nu_mu_cc_p[:,0], self.cs_nu_mu_cc_p[:,1]*1e-38)
        cs_n = np.interp(energy, self.cs_nu_mu_cc_n[:,0], self.cs_nu_mu_cc_n[:,1]*1e-38)
        return 0.5*(cs_p+cs_n)
    def get_cs_nu_mu_bar_cc_iso(self, energy):
        cs_p = np.interp(energy, self.cs_nu_mu_bar_cc_p[:,0], self.cs_nu_mu_bar_cc_p[:,1]*1e-38)
        cs_n = np.interp(energy, self.cs_nu_mu_bar_cc_n[:,0], self.cs_nu_mu_bar_cc_n[:,1]*1e-38)
        return 0.5*(cs_p+cs_n)

    def get_cs_nu_mu_cc_Fe56(self, energy): 
        return 30*self.get_cs_nu_mu_cc_n(energy) + 26*self.get_cs_nu_mu_cc_p(energy)
    def get_cs_nu_mu_bar_cc_Fe56(self, energy): 
        return 30*self.get_cs_nu_mu_bar_cc_n(energy) + 26*self.get_cs_nu_mu_bar_cc_p(energy)
    
    
########################################################################################################################
# 
########################################################################################################################
class make_ATLASdetectorVolume:
    
    def __init__(self, detMass):
        self.detMass = detMass

    massNucleon = 1.67e-27 #in kg
    
    def get_Fe56inHcal(self): 
        return self.detMass / (56*self.massNucleon)


########################################################################################################################
# 
########################################################################################################################
class make_SNeEvent(make_NeutrinoEnsemble, make_SNeNeutrinoSpectrumFromMurase, make_crossSections, make_ATLASdetectorVolume):
    
    def __init__(self, detMass, flux_filename, timerange='all'): 
        
        make_NeutrinoEnsemble.__init__(self, [1,1,1])
        make_ATLASdetectorVolume.__init__(self, detMass)
        make_SNeNeutrinoSpectrumFromMurase.__init__(self, flux_filename)
        make_crossSections.__init__(self)
        
        if timerange=='all':
            self.timeIntegrated_dN_neutrino_dE = self.get_timeIntegrated_dN_neutrino_dE()
        else:
            self.timeIntegrated_dN_neutrino_dE = self.get_timeIntegrated_dN_neutrino_dE_timerange(timerange)
    
    # define the functions necessary to perform the integral
    def interpolate_get_timeIntegrated_dN_neutrino_dE(self, energy):
        return np.interp(np.log10(energy), np.log10(self.Energy_array[0]), self.timeIntegrated_dN_neutrino_dE)
    
    def dN(self, energy, distance): #at high energies such as these the nu_mu cross section can be used for nu_e and nu_tau
        dN = self.get_cs_nu_mu_cc_Fe56(energy) * self.get_Fe56inHcal() * self.interpolate_get_timeIntegrated_dN_neutrino_dE(energy)/(4*np.pi*distance**2) / 2
        dNbar = self.get_cs_nu_mu_bar_cc_Fe56(energy) * self.get_Fe56inHcal() * self.interpolate_get_timeIntegrated_dN_neutrino_dE(energy)/(4*np.pi*distance**2) / 2
        return dN + dNbar
    
    # integrate in log log space
    def get_eventNumber_intLogLogSpace(self, distance, integrationEnergyRange): #energy range in GeV still
        
        x = np.logspace(np.log10(100), np.log10(1e7), num=1000)
        y = self.dN(x, distance)
        integral = integrate_logspace(x, y, integrationEnergyRange[0], integrationEnergyRange[1])[0]
        
        return integral 

########################################################################################################################
# 
########################################################################################################################


########################################################################################################################
# 
########################################################################################################################
class make_statTest():
    
    def __init__(self, signal_array, bkg_array): 
        
        assert len(signal_array)==len(bkg_array),"signal and background must be same size arrays"
        self.signal_array = signal_array
        self.bkg_array = bkg_array
        
    def get_pValue(self):
    
        q0 = 0
    
        for s, b in zip(self.signal_array, self.bkg_array):

            pois = sp.stats.poisson(mu = s)
            N = pois.median() + b
            Y = b
            term = None
            if N != 0:
                term = Y - N + N*np.log(N/Y)
            else:
                term = Y

            q0 = q0 + 2*term

        pval = 0.5*(1 - sp.special.erf(np.sqrt(q0/2)))
        
        return pval
    
