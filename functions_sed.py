#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:05:04 2022

@author: rojas
"""
# Put all required functions for radio, MBB, SED, etc 
import numpy as np
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

def cmtoin(cm):
    #input the desired amount in cm , to be converted to in. and be used in figsize in the future (or something else... who uses inches anymore!!?)
    inch = cm/2.54
    return inch


def freq_mm(freq):
    #freq needs to be given in GHz
    #The output is lambda in mm
    c_kms=const.c.to(u.km/u.s).value
    lamb = c_kms/freq #in microns
    return(lamb/1e3) #in mm

def mm_freq(lamb):
    #lambda in mm
    #freq output will be in GHz
    c_kms=const.c.to(u.km/u.s).value
    freq= c_kms/(lamb*1e3)  #lamb in microns to give GHz
    return(freq) #in GHz

def lum_radio(flux, error, z, dz, alpha, dalpha):
    #Give the value of the flux and error in mJy
    S = flux/1e3 *1e-26 * (u.W/u.m**2/u.Hz) #fro mJy to Jy to SI
    e_S = error/1e3 *1e-26 * (u.W/u.m**2/u.Hz)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
    
    L = (S * 4*np.pi *Dl**2) / ((1+z)**(1+alpha))
    dL_dS = L/S
    dL_dz = L * ((-1-alpha) / (1+z))
    dL_da = -L * np.log(1+z)
    
    err_L = np.sqrt((dL_dS**2 * e_S**2) + (dL_dz**2 * dz**2) + (dL_da**2 * dalpha**2))
    return( L, err_L )



def f_radio(luminosity, error, z, dz, alpha, dalpha):
    #Make sure luminosity and error units are in W/Hz
    L = luminosity*(u.W/u.Hz)
    e_L = error*(u.W/u.Hz)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    dl = cosmo.luminosity_distance(z).to(u.m) #in meters

    S = (L * ((1+z)**(1+alpha))) / (4*np.pi*dl**2) 
    
    dS_dL = S / L
    dS_dz = (S / (1+z)) * (1+alpha)
    dS_da = S * np.log(1+z)
    
    err_S = np.sqrt((dS_dL**2 * e_L**2) + (dS_dz**2 * dz**2) + (dS_da**2 * dalpha**2))
    return( S, err_S )


def fmjy_Lnu(flux, error, z, dz):
    #Make sure input of in mJy
    #output will be luminosity and error in W/Hz

    S = flux/1e3 *1e-26 * (u.W/u.m**2/u.Hz) #fro mJy to Jy to SI
    e_S = error/1e3 *1e-26 * (u.W/u.m**2/u.Hz)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
    
    L = (S * 4*np.pi *Dl**2) / (1+z)
    
    dL_dS = L/S
    dL_dz = -L / (1+z)
    
    err_L = np.sqrt((dL_dS**2 * e_S**2) + (dL_dz**2 * dz**2) )
    
    return( L, err_L )

def Lnu_fmjy(luminosity, error, z, dz):
    #Make sure input of luminosity and error units are in W/Hz
    #output will be in mJy
    L = luminosity*(u.W/u.Hz)
    e_L = error*(u.W/u.Hz)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    dl = cosmo.luminosity_distance(z).to(u.m) #in meters

    S = (L * (1+z)) / (4*np.pi*dl**2) 
    
    dS_dL = S / L
    dS_dz = S / (1+z)
    
    err_S = np.sqrt((dS_dL**2 * e_L**2) + (dS_dz**2 * dz**2))
    
    S = S.to(u.mJy)
    err_S= err_S.to(u.mJy)

    return( S, err_S )

##################
#All possible sync_break models
##################
def sync_break(freq,gamma_fit, gradient_a,nu_break, gradient_break_a):
    #print('nu_break',nu_break)
    y_fit = []
    for nu_i in freq:
        if nu_i <= nu_break:
            y_fit.append(gamma_fit * nu_i**gradient_a)
        elif nu_i > nu_break:
            y_fit.append(gamma_fit * nu_break**(gradient_a - gradient_break_a) * nu_i**gradient_break_a)
            
    return(np.array(y_fit))

def sync_break_CI(freq,gamma_fit, gradient_a,nu_break):
    #print('nu_break',nu_break)
    y_fit = []
    gradient_break_CI = ( gradient_a - (1/2.) )
    for nu_i in freq:
        if nu_i <= nu_break:
            y_fit.append(gamma_fit * nu_i**gradient_a)
        elif nu_i > nu_break:
            y_fit.append(gamma_fit  * nu_break**(gradient_a - gradient_break_CI) * nu_i**gradient_break_CI)
            
    return(np.array(y_fit))


def sync_break_KP(freq,gamma_fit, gradient_a,nu_break):
    #print('nu_break',nu_break)
    y_fit = []
    gradient_break_KP = (((4/3) * gradient_a) -1)
    for nu_i in freq:
        if nu_i <= nu_break:
            y_fit.append(gamma_fit * nu_i**gradient_a)
        elif nu_i > nu_break:
            y_fit.append(gamma_fit  * nu_break**(gradient_a - gradient_break_KP) * nu_i**gradient_break_KP)
            
    return(np.array(y_fit))


##################
#Functions for MBB
##################

#Planck
def planck(T,nu_rest):
    # T = temperature in Kelvin
    #nu_rest = Rest frame frequency in Hz
    T = T * u.K
    nu_rest = nu_rest * u.Hz
    h = const.h
    c = const.c
    k = const.k_B
    cons = 2*h*nu_rest**3 / c**2
    expo = 1/(np.exp((h*nu_rest)/(k*T)) - 1)
    planck = cons*expo

    return planck # units Hz^3 * s^3 * J / m^2 = J/ m^2 == kg / s^2


#Correct for CMB contrast
def f_cmb(z,T,nu_obs,beta= True):
    #T = Temperature in Kelvin ( Intrinsic dust temperature the source would have at z=0.)
    #nu_obs = the range of observed frequencies in GHz for which to calculate CMB contrast
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    #Need to calculate T_cmb
    T_cmb = cosmo.Tcmb(z).value
    
    nu_rest = nu_obs*1e9*(1+z) #convert Obs_Frequency to Hz and to res_freq

    planck_cmbz = planck(T_cmb,nu_rest)

    if beta: #Correct for CMB heating
        T_cmb0 = 2.725
        e = 4+beta
        T_dustz = (T**e + T_cmb0**e * ((1+z)**e -1))**(1/e) 
         #This function shows that the effect of CMB heating is mainly important at higher redshifts
         #I don't need to correct for this parameter for P352-15
         #To prove this, try the same temperature 47K to 30K at different z from 5 to 11
    else:
        T_dustz = T
        
    planck_dustz = planck(T_dustz,nu_rest)
    
        #Now Planck from the 'continuum rest freq'

    f_cmb = 1 - (planck_cmbz/planck_dustz)
    
    return(f_cmb)

def mod_planck_corrected(z, T, nu_obs, beta, Mdust,cmb_heating = True):
    #z = redshift
    #T = Dust temperature in Kelvin
    #nu_obs = observed grequencies in GHz
    #beta = emissivity coefficient usually between 1.6 and 1.95 for high-z quasars.(dimensionless)
    #Mdust = Dust mass previously found in kg.
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
    
    nu_rest = nu_obs*u.GHz * (1+z)
    k_0 = 0.77*(u.cm**2/u.gram)
    kappa = (k_0*((nu_rest/(352*u.GHz))**beta)).to(u.m**2/u.kilogram)#comes from Bram's 2018 paper section 4.3

    # kappa_ref = 2.64*(u.m**2/u.kg)  # m**2/kg
    # kappa_nu_ref = const.c / (125e-6*u.m)  # Hz
    # kappa=kappa_ref * (nu_rest.to(u.Hz) / kappa_nu_ref) ** beta
    
    if cmb_heating == True:
        dust_fobs = f_cmb(z, T, nu_obs,beta) * (1+z) * Dl**-2 * kappa * Mdust *planck(T, nu_rest.to(u.Hz).value).to(u.kg/u.s**2)
    else:
        dust_fobs = f_cmb(z, T, nu_obs) * (1+z) * Dl**-2 * kappa * Mdust *planck(T, nu_rest.to(u.Hz).value).to(u.kg/u.s**2)
    dust_fobs_mjy =  dust_fobs *1e29 #converting from kg/s^2 to mjy
    
    return(dust_fobs_mjy.value)

def MBB_sync_break(x_arr,*labels):
    
        z = x_arr[0]
        Mdust=x_arr[1]
        nu_obs = x_arr[2:]
    
            ### MBB
    
        T = labels[0]
        beta = labels[1]
        
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
        
        nu_rest = nu_obs*u.GHz * (1+z)
        k_0 = 0.77*(u.cm**2/u.gram)
        kappa = (k_0*((nu_rest/(352*u.GHz))**beta)).to(u.m**2/u.kilogram)#comes from Bram's 2018 paper section 4.3
    
        # kappa_ref = 2.64*(u.m**2/u.kg)  # m**2/kg
        # kappa_nu_ref = const.c / (125e-6*u.m)  # Hz
        # kappa=kappa_ref * (nu_rest.to(u.Hz) / kappa_nu_ref) ** beta
        
        dust_fobs = f_cmb(z, T, nu_obs) * (1+z) * Dl**-2 * kappa * Mdust *planck(T, nu_rest.to(u.Hz).value).to(u.kg/u.s**2)
        dust_fobs_mjy =  dust_fobs *1e29 #converting from kg/s^2 to mjy
        #print('DUST+FOBS',dust_fobs_mjy)
        

            ### Synchrotron

        gamma_fit = labels[2]
        gradient_a = labels[3]
        nu_break =labels[4]
        gradient_break_a = labels[5]

        y_fit = sync_break(nu_obs,gamma_fit, gradient_a,nu_break, gradient_break_a)
        #print('Y_FIT',y_fit)
            
        sum_func = dust_fobs_mjy.value + y_fit
        return(sum_func)

def MBB_sync_break_CI(x_arr,*labels):
    
        z = x_arr[0]
        Mdust=x_arr[1]
        nu_obs = x_arr[2:]
    
            ### MBB
    
        T = labels[0]
        beta = labels[1]
        
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
        
        nu_rest = nu_obs*u.GHz * (1+z)
        k_0 = 0.77*(u.cm**2/u.gram)
        kappa = (k_0*((nu_rest/(352*u.GHz))**beta)).to(u.m**2/u.kilogram)#comes from Bram's 2018 paper section 4.3
    
        # kappa_ref = 2.64*(u.m**2/u.kg)  # m**2/kg
        # kappa_nu_ref = const.c / (125e-6*u.m)  # Hz
        # kappa=kappa_ref * (nu_rest.to(u.Hz) / kappa_nu_ref) ** beta
        
        dust_fobs = f_cmb(z, T, nu_obs) * (1+z) * Dl**-2 * kappa * Mdust *planck(T, nu_rest.to(u.Hz).value).to(u.kg/u.s**2)
        dust_fobs_mjy =  dust_fobs *1e29 #converting from kg/s^2 to mjy
        #print('DUST+FOBS',dust_fobs_mjy)
        

            ### Synchrotron

        gamma_fit = labels[2]
        gradient_a = labels[3]
        nu_break =labels[4]
        y_fit = sync_break_CI(nu_obs,gamma_fit, gradient_a,nu_break)
        #print('Y_FIT',y_fit)
            
        sum_func = dust_fobs_mjy.value + y_fit
        return(sum_func)


def MBB_sync_break_KP(x_arr,*labels):
    
        z = x_arr[0]
        Mdust=x_arr[1]
        nu_obs = x_arr[2:]
    
            ### MBB
    
        T = labels[0]
        beta = labels[1]
        
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        Dl = cosmo.luminosity_distance(z).to(u.m) #in meters
        
        nu_rest = nu_obs*u.GHz * (1+z)
        k_0 = 0.77*(u.cm**2/u.gram)
        kappa = (k_0*((nu_rest/(352*u.GHz))**beta)).to(u.m**2/u.kilogram)#comes from Bram's 2018 paper section 4.3
    
        # kappa_ref = 2.64*(u.m**2/u.kg)  # m**2/kg
        # kappa_nu_ref = const.c / (125e-6*u.m)  # Hz
        # kappa=kappa_ref * (nu_rest.to(u.Hz) / kappa_nu_ref) ** beta
        
        dust_fobs = f_cmb(z, T, nu_obs) * (1+z) * Dl**-2 * kappa * Mdust *planck(T, nu_rest.to(u.Hz).value).to(u.kg/u.s**2)
        dust_fobs_mjy =  dust_fobs *1e29 #converting from kg/s^2 to mjy
        #print('DUST+FOBS',dust_fobs_mjy)
        

            ### Synchrotron

        gamma_fit = labels[2]
        gradient_a = labels[3]
        nu_break =labels[4]
        y_fit = sync_break_KP(nu_obs,gamma_fit, gradient_a,nu_break)
        #print('Y_FIT',y_fit)
            
        sum_func = dust_fobs_mjy.value + y_fit
        return(sum_func)


######################
#find Synchrotron age
######################

def t_syn(B,nu_b,err_B,err_nu_b):
    #B = Magnetic field in  microGauss
    #nu_b = rest frame frequency break in GHz
    #Output is in Myr
    tsyn = 1610 * B**(-3/2) * nu_b**(-1/2)
    
    #Error in the calculation
    dtsyn_dnu = 1610* B**(-3/2) *((-1/2)*nu_b**(-3/2))
    dtsyn_dB = 1610* nu_b**(-1/2) *((-3/2)*nu_b**(-5/2))
    err_tsyn = np.sqrt((dtsyn_dnu**2 * err_nu_b**2) + (dtsyn_dB**2 * err_B**2))
    return(tsyn,err_tsyn)

def t_spec(B,nu_b,B_cmb,err_B,err_nu_b,err_B_cmb):
    #B = Magnetic field in  nT. 1Gauss = 10e-4T or 1 Gauss=10e5 nT
    #nu_b = rest frame frequency break in GHz.
    #B_cmb = equivalent magnetic field to the CMB’s energy density in nanoT
    #Output is in Myr
    tspec = 50.3 * (B**(1/2)/(B**2 + B_cmb**2)) * nu_b**(-1/2)
    
    #Error in the calculation
    dtspec_dnu = -25.15*nu_b**(-3/2)*( B**(1/2) / (B**2 + B_cmb**2))
    dtspec_dB = -25.15*nu_b**(-1/2)*B**(-1/2)*(((3*B**2) - B_cmb**2)/(B**2+B_cmb**2)**2)
    dtspec_dBcmb = -100.6*nu_b**(-1/2)*((B**(1/2)* B_cmb) / (B**2+B_cmb**2)**2)
    err_tspec = np.sqrt((dtspec_dnu**2 * err_nu_b**2) + (dtspec_dB**2 * err_B**2)+ (dtspec_dBcmb**2 * err_B_cmb**2))
    return(tspec,err_tspec)
