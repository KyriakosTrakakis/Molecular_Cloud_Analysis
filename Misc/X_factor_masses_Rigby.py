#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:50:04 2025

@author: kyriakos.trakakis
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, join
 
plt.ion()
 
cat1 = Table.read('/Users/kyriakos.trakakis/Downloads/CHIMPS_13CO_source_properties.fits')
cat2 = Table.read('/Users/kyriakos.trakakis/Downloads/CHIMPS_13CO_physical_properties.fits')
 
cata = join(cat1, cat2)
 
mass = cata['Mass'].data.data * u.Msun
eta_mb = 0.72
dv = 0.5 * u.km / u.s
mu = 2.8
Xco = 6E21 * u.cm**-2 / (u.K * u.km / u.s)  # From Rigby et al. 2025
dist = cata['Dist'].data.data * u.kpc
Wco = cata['Sum_13CO'].data * u.K * dv / eta_mb # Integrated intensity
pixsize = (7.612 * u.arcsec).to('rad').value
massX = mu * c.m_p * dist**2 * Xco * Wco * pixsize**2
massX = massX.to('Msun')
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(np.log10(massX/mass), bins=50)
ax.set_xlabel('log10(Mass_R+25 / Mass_R+19 (Rigby+19))')
