from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from  astropy.coordinates import cartesian_to_spherical as cart2sphere
import matplotlib.pyplot as plt
import healpy as hp
from scipy.spatial.transform import Rotation as R

from estoiles import gw_calc as gwc
from estoiles import calc_dn as cdn
from estoiles import gw_source as gws


def nan_if(arr, value):
   return np.where(arr == value, np.nan, arr)  

@dataclass
class GW_cls:
   nside: int = 64
   min_the: float = 0
   max_the: float = np.pi
   min_phi: float = 0
   max_phi: float = 2*np.pi
   do_plot: bool = True
   lmax_plot: int = 12
   filename: str = 'cl_output.png'
   pi = np.pi   
   gw_source: gws = gws.GWSource(freq=1.e-6*u.Hz)
   
   def __post_init__(self):
      print("starting computation")
      self.create_map()
      self.hcalc()
      self.dncalc()

      self.plot_map(self.data[0])
      if self.max_the == np.pi and self.min_the==0 and self.max_phi ==2*np.pi and self.min_phi ==0:
         pass
      else:
         self.mask_map()
      self.mean_subtract()
      self.get_cls()
      print("Cls computed")
      if self.do_plot:
         self.plot_cls()
   
   def create_map(self):
      npix = hp.nside2npix(self.nside)
      self.THE,self.PHI = hp.pix2ang(self.nside, ipix = range(npix))
     
      
   def mask_map(self):
      for i, j, idx in zip(self.THE, self.PHI, range(len(self.THE))):
         if i<self.min_the or i>self.max_the or j<self.min_phi or j>self.max_phi:
            self.data[0][idx] = -1.6375e+30 # Setting data to a noticeable value
            self.data[1][idx] = -1.6375e+30
            self.data[0][idx] = hp.UNSEEN # To eventually create new HealPy map
            self.data[1][idx] = hp.UNSEEN
            
   def hcalc(self):
      self.hmat = self.gw_source.h
      
   def dncalc(self):
      sin = np.sin
      cos = np.cos
      self.q = self.gw_source.srcindet
      n = np.array([self.THE, self.PHI])
      starcoord = np.array([sin(self.THE*u.rad)*cos(self.PHI*u.rad),sin(self.THE*u.rad)*sin(self.PHI*u.rad),cos(self.THE*u.rad)])
      self.data_cartesian = cdn.dn(self.hmat, self.q, starcoord)
      self.Mat = np.array([[sin(self.THE)*cos(self.PHI),sin(self.THE)*sin(self.PHI),cos(self.THE)],
                    [cos(self.THE)*cos(self.PHI),cos(self.THE)*sin(self.PHI),-sin(self.THE)],
                    [-sin(self.THE),cos(self.THE),np.zeros_like(self.PHI)]])
      dn_sph = np.einsum('ijk,jk->ik',self.Mat,self.data_cartesian)      
      self.data = dn_sph[1:]
      
   def mean_subtract(self):
      self.data -= np.nanmean(nan_if(self.data, -1.6375e+30),axis=1).reshape([2,1])
      self.data_ma = hp.ma(self.data)      
      
   def get_cls(self):
      self.alm_hp = np.array(hp.sphtfunc.map2alm_spin(self.data_ma, spin = 1))
      self.Cl_hp = np.array(hp.alm2cl(self.alm_hp))      
   def plot_cls(self):
      import os
      import matplotlib
      from matplotlib import rc
      os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2020/bin/x86_64-darwin'
      from matplotlib import rc
      plt.rc('text', usetex=True)
      plt.rc('xtick', labelsize=20)
      plt.rc('ytick', labelsize=20)
      plt.rc('axes', labelsize=22)
      plt.rc('legend', fontsize=16)
      plt.rc('axes', titlesize=22)
      params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
      plt.rcParams.update(params)      
      lmax = self.lmax_plot
      fig = plt.figure()
      plt.plot(np.arange(1,lmax+1), np.sqrt(self.Cl_hp[0][1:lmax+1:]),marker='.',label=r'E')
      plt.plot(np.arange(1,lmax+1), np.sqrt(self.Cl_hp[1][1:lmax+1:]),marker='.',label=r'B')
      plt.title(r'$\mathrm{Power \ Spectrum}$')
      plt.ylabel(r'$\sqrt{C_\ell}$')
      plt.xlabel(r'$\ell$')
      #plt.yscale('log')
      plt.legend()
      plt.tight_layout()
      plt.savefig(self.filename, dpi = 300)
   def plot_map(self, data):
      import os
      import matplotlib
      from matplotlib import rc
      os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2020/bin/x86_64-darwin'
      from matplotlib import rc
      plt.rc('text', usetex=True)
      plt.rc('xtick', labelsize=20)
      plt.rc('ytick', labelsize=20)
      plt.rc('axes', labelsize=22)
      plt.rc('legend', fontsize=16)
      plt.rc('axes', titlesize=22)
      params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
      plt.rcParams.update(params)  
      fig = plt.figure()
      hp.mollview(data)
      plt.savefig('hp_map.png', dpi = 300)

      
            
      
