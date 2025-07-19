'''
This script generates interpolated waveform functions
'''

import numpy as np
from estoiles import gw_calc as gwc
from estoiles import calc_dn as cdn
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy.optimize

c = 299792458*u.m/u.s
G = 6.67e-11*u.m**3/u.kg/u.s**2
s_mass=G*(1.98892*10**(30)*u.kg)/(c**3)

freq = 5.e-8*u.Hz
inc = 0.*u.deg
psi = 0.*u.deg
phip = 0.*u.deg
phic = 0.*u.deg
scoord = SkyCoord(l = 90*u.deg, b=90*u.deg, frame='galactic')
telcoord = SkyCoord(l = 0*u.deg, b=0*u.deg, frame='galactic')

def h(theta_,OL):
    '''Compute the h tensor.

    Keyword Parameters:
    theta_ -- (Nsample,Nvar)-dimensional parameter array
    OL -- PN order, boolean array

    Returns:
    hMat -- (3,3,Nsample)-dimensional array
    '''

    hMat = np.empty([3,3,1])
    for i in range(0,theta_.shape[0]):
        logMc,logq,logd = theta_[i,:]
        g = gwc.GWcalc(10.**logMc*u.Msun,10.**logq, freq, 10.**logd*u.Mpc, inc, psi, scoord, telcoord)
        theta = 0*u.Myr,OL
        h = (g.calc_h(theta,phi_=0*u.deg)).reshape([3,3,1])
        hMat = np.append(hMat,h,axis = 2)
    return hMat[:,:,1:]

def Mc(q_,f_):
    return (10**q_/(1+10**q_)**2)**.6/(f_.to(1/u.s)*6.**1.5*np.pi*G/c**3).to(1/u.Msun)

PV = np.empty([3,2])
PV[2,0] = -5.
PV[2,1] = 2.
PV[1,0] = -2.
PV[1,1] = 0.
PV[0,1] = np.log10(Mc(PV[1,0],freq).value)
PV[0,0] = PV[0,1] - 4.

Nmc,Nq,NR = 50,50,50
grid_logMc,grid_logq,grid_logR = np.mgrid[PV[0,0]:PV[0,1]:Nmc*1j, PV[1,0]:PV[1,1]:Nq*1j,PV[2,0]:PV[2,1]:NR*1j]
Np = Nmc*Nq*NR
a = grid_logMc.reshape([Np,1])
b = grid_logq.reshape([Np,1])
c = grid_logR.reshape([Np,1])
GridMatrix = np.hstack((a,b,c))

FileName = 'PNInterpolator/h1PN'
orderList = np.array([True,True,True,False,False])
Grid_h = h(GridMatrix,orderList)
hmodel = Grid_h.reshape(3,3,Nmc,Nq,NR)
np.save(FileName,hmodel.value)
GM = GridMatrix.reshape([Nmc,Nq,NR,3])
np.save(FileName+'_Grid',GM)

McList = GM[:,0,0,0].flatten()
qList = GM[0,:,0,1].flatten()
RList = GM[0,0,:,2].flatten()
paramList = (McList,qList,RList)

h_func = []
for i in range(0,3):
    for j in range(0,3):
        h_func.append(RegularGridInterpolator(paramList,hmodel[i,j,:,:,:]))
np.save(FileName+'_func',h_func)
