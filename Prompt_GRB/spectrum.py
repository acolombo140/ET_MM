import numpy as np

keV_to_erg = 1.60217662e-09

def dNdE_comp(E,Ep,alpha,beta):
    """ 
    GRB "comptonized" spectrum (power law with an exponential cut-off).
    
    Parameters:
    - E: photon energy
    - Ep: SED peak energy
    - alpha: power-law slope
    - beta: not used
    
    E and Ep must have the same units.
    
    Returns: dNdE
    - dNdE: (un-normalized) photon flux density at E
    """
    
    return E**alpha*np.exp(-(alpha+2.)*E/Ep) 

def dNdE_Band(E,Ep,alpha,beta):
    """ 
    GRB "Band" spectrum (two power laws with an exponential transition).
    
    Parameters:
    - E: photon energy
    - Ep: SED peak energy
    - alpha: low-energy power-law slope
    - beta: high-energy power-law slope
    
    E and Ep must have the same units.
    
    Returns: dNdE
    - dNdE: (un-normalized) photon flux density at E
    """
    
    Ec = ((alpha-beta)*Ep/(alpha+2.))
    
    dnde = E**alpha*np.exp(-(alpha+2.)*E/Ep)
    dnde[E>Ec]=E[E>Ec]**beta*np.exp(beta-alpha)*Ec**(alpha-beta)
    
    return dnde


def photon_flux(Liso,Ep,alpha,z,dL,band=[10.,1000.],beta=-2.3,model='Comp'):
    """ 
    GRB photon flux in a detector.
    
    Parameters:
    - Liso: isotropic equivalent luminosity [erg/s]
    - Ep: SED peak energy [keV]
    - alpha: low-energy power-law slope
    - z: redshift
    - dL: luminosity distance [cm]
    
    Keywords:
    - band: tuple containing the low and high ends of the detector band in keV
    - beta: the high-energy spectral slope, if applicable
    - model: the spectrum model (either 'Comp' or 'Band')
    
    Returns: P
    - P: photon flux in ph cm-2 s-1
    """
    
    if model=='Band' or model=='band':
        dNdE = dNdE_Band
    else:
        dNdE = dNdE_comp
    
    E = np.logspace(np.log10(Ep)-2,np.log10(Ep)+2,1000) #-1
    #E = np.logspace(np.log10(1+z),np.log10(1.e4*(1+z)),1000) #-1
    F = np.trapz(E**2*dNdE(E,Ep/(1+z),alpha,beta),np.log(E))*keV_to_erg #(1+z)*Ep
    
    E = np.logspace(np.log10(band[0]),np.log10(band[1]),300)
    P = Liso/(4*np.pi*dL**2*F)*np.trapz(E*dNdE(E,Ep/(1+z),alpha,beta),np.log(E)) 
    
    return P
    

