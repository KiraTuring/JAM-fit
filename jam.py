import numpy as np
import matplotlib.pyplot as plt
import json
import sys, getopt

import jampy as jam 
from adamet.adamet import adamet
from adamet.corner_plot import corner_plot
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from mgefit.mge_fit_1d import mge_fit_1d

from astropy.cosmology import FlatLambdaCDM

def dark_halo_mge(gamma, rbreak, rmin=1, rmax=100):
    """
    Returns the MGE parameters for a generalized NFW dark halo profile
    https://ui.adsabs.harvard.edu/abs/2001ApJ...555..504W
    - gamma is the inner logarithmic slope (gamma = -1 for NFW)
    - rbreak is the break radius in arcsec

    """
    n = 300     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(rmin, rmax, n)   # logarithmically spaced radii in arcsec
    rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    m = mge_fit_1d(r, rho, ngauss=30, quiet=1, plot=0, inner_slope=-gamma, outer_slope=3)

    surf_dm, sigma_dm = m.sol           # Total count and sigma
    surf_dm /= sigma_dm*np.sqrt(2*np.pi) # Convert to Peak surface density 
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm


def mge1d_plot(surf, sigma, q_obs, start=0.1, end=100, num=300, plot=False, **kwargs):
    radi = np.geomspace(start,end,num)
    mgesurf = np.zeros([2,len(radi)])

    for i,f in enumerate(surf):
        sig = sigma[i]
        q = q_obs[i]
        fa = f*np.exp(-(radi/sig)**2/2)
        fb = f*np.exp(-(radi/sig/q)**2/2)
        if plot:
            plt.plot(radi,fa,label=i,alpha=0.5,color='k')
        mgesurf[0] += fa
        mgesurf[1] += fb

    plt.plot(radi, mgesurf.T, **kwargs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('r/arcsec')
    plt.ylabel(r'$I/L_\odot pc^{-2}$')

    return radi, mgesurf[0], mgesurf[1]


def total_mass_mge(surf_lum, sigma_lum, qobs_lum, gamma, rbreak, f_dm, inc):
    """
    Combine the MGE from a dark halo and the MGE from the stellar surface
    brightness in such a way to have a given dark matter fractions f_dm
    inside a sphere of radius one half-light radius reff

    """
    surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak, 0.1*np.min(sigma_lum), 10*np.max(sigma_lum))

    reff = jam.mge.half_light_radius(surf_lum, sigma_lum, qobs_lum)[0]
    stars_lum_re = jam.mge.radial_mass(surf_lum, sigma_lum, qobs_lum, inc, reff)
    dark_mass_re = jam.mge.radial_mass(surf_dm, sigma_dm, qobs_dm, inc, reff)

    # Find the scale factor needed to satisfy the following definition
    # f_dm == dark_mass_re*scale/(stars_lum_re + dark_mass_re*scale)
    scale = (f_dm*stars_lum_re)/(dark_mass_re*(1 - f_dm))

    surf_pot = np.append(surf_lum, surf_dm*scale)   # Msun/pc**2. DM scaled so that f_DM(Re)=f_DM
    sigma_pot = np.append(sigma_lum, sigma_dm)      # Gaussian dispersion in arcsec
    qobs_pot = np.append(qobs_lum, qobs_dm)

    return surf_pot, sigma_pot, qobs_pot


def jam_lnprob(pars, surf_lum=None, sigma_lum=None, qobs_lum=None,
              surf_pot=None, sigma_pot=None, qobs_pot=None, dist=None, 
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None, 
              rms=None, erms=None, pixsize=None, plot=True, align='cyl',
              pars0=None, idfit=None):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    pars0[idfit] = pars
    q, ratio, lg_mbh, lg_ml, f_dm, lg_rb, psfscale = pars0

    sigmapsf[0] *= psfscale
    
    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    if 0 == f_dm:
        # Here assume mass follows light
        surf_pot = surf_lum
        sigma_pot = sigma_lum
        qobs_pot = qobs_lum

    else:
    # # These parameters could be fitted from good and spatially-extended data.
        gamma = -1                  # Adopt fixed NFW inner halos slope
        rbreak = 10**lg_rb               # in arcsec
    # pc = distance*np.pi/0.648   # Constant factor to convert arcsec --> pc
    # rbreak /= pc                # Convert the break radius from pc --> arcsec
        surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, gamma, rbreak, f_dm, inc)

    
    # Note: surf_pot is multiplied by ml, while I set the keyword ml=1
    surf_pot_true = np.array(surf_pot)*10**lg_ml

    out = jam.axi.proj(surf_lum, sigma_lum, qobs_lum, surf_pot_true, sigma_pot, qobs_pot,
                       inc, 10**lg_mbh, dist, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1)

    # These two lines are just for the final plot
    jam_lnprob.rms_model = out.model
    jam_lnprob.flux_model = out.flux

    resid = (rms[goodbins] - out.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

if __name__ == '__main__':

    path = None
    align = 'cyl'

    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:s",["path=","sph"])
    except getopt.GetoptError:
        print('test.py -p <path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--path"):
            path = arg
        elif opt in ("-s", "--sph"):
            align = 'sph'

    if path == None:
        print('Error: No file path')
        sys.exit(2)

    with open('%s/jam_pars.json'%path) as f:
        param_dict = json.load(f)
    globals().update(param_dict)

    print('Dir:',path)

    print('Distance: %.2f Mpc'%distance)
    print('Scale: %.2f pc/arcsec'%(distance/206265*1e6))

    surf, sigma, qobs = np.loadtxt('%s/mge_parms.dat'%path).T
    # print(surf, sigma, qObs)

    xbin, ybin, rms, erms, goodbins = np.loadtxt('%s/kinematics.dat'%path).T
    goodbins = goodbins==1
    p0 = np.array(p0)

    kwargs = {'surf_lum': surf, 'sigma_lum': sigma, 'qobs_lum': qobs,
            'dist': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
            'normpsf': normpsf, 'rms': rms, 'erms': erms, 'pixsize': pixsize,
            'goodbins': goodbins, 'plot': 0, 'align':align, 'pars0':p0.copy(), 'idfit':idfit}
    
    print('#iteration', nstep)
    print("Started AdaMet please wait...")
    init = p0[idfit]
    sigpar = np.array(sigpar)[idfit]
    bounds = np.array(bounds)[:,idfit]
    labels = np.array(labels)[idfit]

    pars, lnprob = adamet(jam_lnprob, init, sigpar, bounds, nstep,
                        kwargs=kwargs, nprint=nstep/20, labels=labels, seed=3)
    

    corner_plot(pars, lnprob, labels=labels, extents=bounds)
    # plt.colorbar()
    plt.savefig('%s/corner.png'%path)

    np.savetxt('%s/jam_lnprob.dat'%path, np.hstack([pars, np.transpose([lnprob])]), header='%s, lnprob'%labels)
    np.savetxt('%s/model.dat'%path, np.vstack([xbin, ybin, jam_lnprob.rms_model, jam_lnprob.flux_model]).T, fmt='%10.5f', header=' xbin | ybin | Vrms_model | flux_model')
    