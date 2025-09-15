import numpy as np
from . import spectrum

# isotropic equivalent radiated energy from a structured jet towards an observer at a given angle, following Salafia et al. 2015
def Eiso_structured(tv,theta,dE_dOmega,Gamma,phi_res=100):
    """
    Isotropic equivalent radiated energy from a structured jet towards an observer at a given angle, following Salafia et al. 2015
    
    Parameters:
    -----------
    - tv: viewing angle in radians
    - theta: array of angles (of length N, in radians) over which the jet structure is defined
    - dE_dOmega: array of radiated energy per jet unit solid angle (length N, in erg/sr) evaluated at theta
    - Gamma: array of jet Lorentz factors (length N) evaluated at theta
    
    Keywords:
    ---------
    - phi_res: azimuthal angular resolution (int). The polar resolution is set by the length of the theta array.
    
    Returns: Eiso(thv), the isotropic-equivalent energy as seen by the observer, in erg
    
    """
    
    phi = np.linspace(0.,2*np.pi,phi_res)
    
    TH,PHI = np.meshgrid(theta,phi)
    
    cosa = np.cos(TH)*np.cos(tv)+np.sin(PHI)*np.sin(TH)*np.sin(tv)
    
    G,dummy = np.meshgrid(Gamma,phi)
    e,dummy = np.meshgrid(dE_dOmega,phi)
    
    b = (1.-G**-2)**0.5
    
    delta = (G*(1.-b*cosa))**-1
    
    f = delta**3*e/G
    
    return np.trapz(np.trapz(f*np.sin(TH),theta,axis=1),phi,axis=0)

def Ep_structured(Epc,tv,theta,dE_dOmega,Gamma,phi_res=100):
    """
    Peak photon energy in the SED of the prompt emission from a structured jet, as measured by an observer at a given angle, following Salafia et al. 2015
    
    Parameters:
    -----------
    - Epc: comoving spectral peak photon energy, in keV (assumed independent of angle)
    - tv: viewing angle in radians
    - theta: array of angles (of length N, in radians) over which the jet structure is defined
    - dE_dOmega: array of radiated energy per jet unit solid angle (length N, in erg/sr) evaluated at theta
    - Gamma: array of jet Lorentz factors (length N) evaluated at theta
    
    Keywords:
    ---------
    - phi_res: azimuthal angular resolution (int). The polar resolution is set by the length of the theta array.
    
    Returns: Ep(thv), in keV
    
    """
    
    phi = np.linspace(0.,2*np.pi,phi_res)
    
    TH,PHI = np.meshgrid(theta,phi)
    
    cosa = np.cos(TH)*np.cos(tv)+np.sin(PHI)*np.sin(TH)*np.sin(tv)
    
    G,dummy = np.meshgrid(Gamma,phi)
    e,dummy = np.meshgrid(dE_dOmega,phi)
    
    b = (1.-G**-2)**0.5
    
    delta = (G*(1.-b*cosa))**-1
    
    ep = Epc/G*delta**4*e/G
    f = delta**3*e/G
    
    return np.trapz(np.trapz(ep*np.sin(TH),theta,axis=1),phi,axis=0)/np.trapz(np.trapz(f*np.sin(TH),theta,axis=1),phi,axis=0)

def photon_flux_structured(Epc,T,z,dL,tv,theta_0,dE_dOmega,Gamma,alpha=-0.86,beta=-2.3,band=[10., 1000.],model='Comp',phi_res=100,Gamma_min=10.):
    """
    Photon flux in a given band, associated to the prompt emission from a structured jet, as seen by an observer at a given angle, following Salafia et al. 2015
    
    Parameters:
    -----------
    - Epc: comoving spectral peak photon energy, in keV (assumed independent of angle)
    - T: emission duration, in seconds (assumed independent of angle)
    - z: redshift
    - dL: luminosity distance, in cm
    - tv: viewing angle in radians
    - theta_0: array of angles (of length N, in radians) over which the jet structure is defined
    - dE_dOmega: array of radiated energy per jet unit solid angle (length N, in erg/sr) evaluated at theta
    - Gamma: array of jet Lorentz factors (length N) evaluated at theta (must be monotonically decreasing)
    
    Keywords:
    ---------
    - alpha: low-energy spectral index. Default: -0.86 (the average value for Fermi GRBs, Nava et al. 2012)
    - beta: high-energy spectral index (has no effect if the COMP spectrum is selected). Default: -2.3 (the average value for Fermi GRBs)
    - band: list with two elements, namely the low- and high- end of the spectral band, in keV. Default: [10., 1000.]
    - model: either 'Comp' (power law of index alpha, with exponential cutoff at Epc), or 'Band' (the Band function, i.e. two power laws smoothly connected by an exponential - Band et al. 1993)
    - phi_res: azimuthal angular resolution (int). The polar resolution is set by the length of the theta array.
    - Gamma_min: Lorentz factor below which no prompt emission is produced
    
    Returns: Eiso(thv), the isotropic-equivalent energy as seen by the observer, in erg
    
    """
    
    
    phi = np.linspace(0.,2*np.pi,phi_res)
    theta = theta_0[Gamma>Gamma_min]
    
    TH,PHI = np.meshgrid(theta,phi)
    
    cosa = np.cos(TH)*np.cos(tv)+np.sin(PHI)*np.sin(TH)*np.sin(tv)
    
    G,dummy = np.meshgrid(Gamma[Gamma>Gamma_min],phi)
    e,dummy = np.meshgrid(dE_dOmega[Gamma>Gamma_min],phi)
    
    b = (1.-G**-2)**0.5
    
    delta = (G*(1.-b*cosa))**-1
    
    ep = Epc/2./G[0,0]*delta
    l = delta**3*e/G/T
    
    dp = np.zeros(G.shape)
    
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            dp[i,j] = spectrum.photon_flux(l[i,j],ep[i,j],alpha,z,dL,band=band,beta=beta,model=model)
        
    return np.trapz(np.trapz(dp*np.sin(TH),theta,axis=1),phi,axis=0)


def dEiso_dnu_structured(nu,tv,theta,dE_dOmega,Gamma,nu_comov,fnu_comov,phi_res=100):
    """
    Isotropic equivalent radiated energy per unit frequency from a structured jet towards an observer at a given angle, following Salafia et al. 2015
    
    Parameters:
    -----------
    - tv: viewing angle in radians
    - theta: array of angles (of length N, in radians) over which the jet structure is defined
    - dE_dOmega: array of radiated energy per jet unit solid angle (length N, in erg/sr) evaluated at theta
    - Gamma: array of jet Lorentz factors (length N) evaluated at theta
    - nu_comov: an array of comoving frequencies (array of length M, in Hz)
    - fnu_comov: a comoving spectral shape (array of length M, arbitrary normalization)
    
    Keywords:
    ---------
    - phi_res: azimuthal angular resolution (int). The polar resolution is set by the length of the theta array.
    
    Returns: Eiso(thv), the isotropic-equivalent energy as seen by the observer, in erg
    
    """
    
    # normalize fnu to one
    fnu = fnu_comov/np.trapz(nu_comov*fnu_comov,np.log(nu_comov))
    
    phi = np.linspace(0.,2*np.pi,phi_res)
    
    TH,PHI,NU = np.meshgrid(theta,phi,nu)
    
    cosa = np.cos(TH)*np.cos(tv)+np.sin(PHI)*np.sin(TH)*np.sin(tv)

    
    G,dummy,dummy2 = np.meshgrid(Gamma,phi,nu)
    e,dummy,dummy2 = np.meshgrid(dE_dOmega,phi,nu)
    
    b = (1.-G**-2)**0.5
    
    delta = (G*(1.-b*cosa))**-1
    
    f = delta**2*e/G
    
    return np.trapz(np.trapz(np.interp(NU/delta,nu_comov,fnu,right=0.,left=0.)*f*np.sin(TH)*TH,np.log(theta),axis=1),phi,axis=0)


if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize']=(3.5,3.)
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['font.size']=11.
    plt.rcParams['font.family']='Liberation Serif'
    
    
    # off-axis apparent structure figures
    
    ## uniform jet
    Eiso_onaxis = 1e51
    thc = 5./180.*np.pi
    Ec = Eiso_onaxis/(4*np.pi)
    Gc = 100.
    th = np.linspace(0.,5*thc,300)
    e = np.full_like(th,Ec)
    e[th>thc]=0.
    g = np.full_like(th,Gc)
    
    thv = np.logspace(0,2.,100)/180.*np.pi

    Eiso100 = np.zeros(100)
    Eiso30 = np.zeros(100)
    
    for i,tv in enumerate(thv):
        Eiso100[i] = Eiso_structured(tv,th,e,g)
        Eiso30[i] = Eiso_structured(tv,th,e,g*3/10)
    
    plt.loglog(th/np.pi*180,e*(4*np.pi),'-',lw=2,color='grey')
    
    plt.xlabel(r'$\theta_\mathrm{view}$ or $\theta$ [deg]')
    plt.ylabel(r'$4\pi \, dE/d\Omega(\theta)$ [erg]')
    
    plt.ylim([1e43,2e51])
    
    plt.twinx()
    
    plt.loglog(thv/np.pi*180,Eiso30,'-b',lw=2)
    plt.loglog(thv/np.pi*180,Eiso100,'-r',lw=2)
    
    plt.ylabel(r'$E_\mathrm{iso}(\theta_\mathrm{view})$ [erg]')
    
    plt.ylim([1e43,2e51])
    plt.xlim([1.,90.])
    
    plt.annotate('$\Gamma=30$',xy=(10.,2e48),rotation=-50)
    plt.annotate('$\Gamma=100$',xy=(10.,2e46),rotation=-50)
    
    plt.savefig('Eiso_thv_uniform.pdf')
    
    plt.clf()

    ## Gaussian jet
    Eiso_onaxis = 1
    thc = 0.1
    Ec = Eiso_onaxis/(4*np.pi)
    Gc = 100.
    th = np.linspace(0.,np.pi/2.,300)
    thg = 0.2
    e = Ec*np.exp(-(th/thc)**2)
    g = (Gc-1.)*np.exp(-(th/thg)**2)+1.
    
    thv = np.logspace(0,2,100)/180.*np.pi

    Eiso = np.zeros(100)
    
    for i,tv in enumerate(thv):
        Eiso[i] = Eiso_structured(tv,th,e,g)
    
    # energy structure
    plt.loglog(th/np.pi*180,e*(4*np.pi),'--',lw=2,color='grey',label='kinetic')
    
    # Lorentz factor
    ax = plt.gca()
    plt.twinx()
    plt.loglog(th/np.pi*180,g,'-b',lw=2)
    plt.ylabel(r'$\Gamma(\theta)$')
    plt.ylim([1,120])
    plt.xlim([1.,90.])
    
    plt.sca(ax)
    
    plt.xlim([1.,90.])
    
    plt.loglog(thv/np.pi*180,Eiso,'-r',lw=2,label='radiated')
    plt.plot([0],[0],'-b',label='Lorentz factor')
    
    plt.xlabel(r'$\theta_\mathrm{view}$ or $\theta$ [deg]')
    plt.ylabel(r'Isotropic equivalent energy')
    
    ylim = plt.ylim([1e-10,2e0])
    
    
    plt.legend(frameon=False,fontsize=10)
    
    
    
    
    plt.savefig('Eiso_thv_Gaussian.pdf')
    
    plt.clf()
    
    #exit()
    
    # off-axis photon flux figure
    
    dL = 230*3.08e24
    z = 0.054
    
    Epc = 600.
    
    Tjet = 0.1
    
    Ekjet = 3.4e49
    thc = 0.1
    Ec = Ekjet/(np.pi*thc**2)
    
    eta = 0.1
    
    Gc = 100.
    thg = 0.2
    
    th = np.linspace(0.,5*thc,60)

    e = eta*Ec*np.exp(-(th/thc)**2)

    g = (Gc-1.)*np.exp(-(th/thg)**2)+1.
    
    thv = np.linspace(0.,60.,30)/180*np.pi
    
    Eiso = np.zeros(30)
    Ep = np.zeros(30)
    
    for i,tv in enumerate(thv):
        Eiso[i] = Eiso_structured(tv,th,e,g)
        Ep[i] = Ep_structured(Epc,tv,th,e,g)
    
    plt.semilogy(thv/np.pi*180,Eiso,'-b')
    
    plt.ylabel(r'$E_\mathrm{iso}\,\mathrm{[erg]}$')
    plt.xlabel(r'$\theta_\mathrm{view}\,\mathrm{[deg]}$')
    
    plt.xlim([0.,60.])
    
    Liso = Eiso/Tjet
    
    P = np.zeros(30)
    PGBM = np.zeros(30)
    PSwift = np.zeros(30)
    
    for i in range(len(Liso)):
        #P[i] = spectrum.photon_flux(Liso[i],Ep[i],-0.86,z,dL)
        PGBM[i] = photon_flux_structured(Epc,Tjet,z,dL,thv[i],th,e,g,phi_res=30)
        PSwift[i] = photon_flux_structured(Epc,Tjet,z,dL,thv[i],th,e,g,band=[15.,150.],phi_res=30)
    
    plt.twinx()
    
    #plt.semilogy(thv/np.pi*180,P,'--b')
    plt.semilogy(thv/np.pi*180,PGBM,'-',color='orange')
    plt.semilogy(thv/np.pi*180,PSwift,'--',color='red')
    
    plt.plot([0.,60.],[0.2,0.2],'--k')
    plt.annotate('GBM limit',xy=(40.,0.07))
    plt.plot([0.,60.],[0.4,0.4],':k')
    plt.annotate('BAT limit',xy=(40.,0.49))
    
    plt.ylabel(r'$P_\mathrm{[h\nu_0,h\nu_1]}\,\mathrm{[ph\,cm^{-2}\,s^{-1}]}$')
    
    plt.savefig('Eiso_Pflux_thv.pdf')
    
    plt.show()
    
    plt.semilogy(thv/np.pi*180,Ep,'--b')
    plt.show()
    
    
