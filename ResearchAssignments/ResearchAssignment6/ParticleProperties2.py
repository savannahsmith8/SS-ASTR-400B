# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:26:19 2025

@author: savan
"""

import numpy as np
import astropy.units as u
from ReadFile import Read

def ParticleProperties(filename, particle_type, particle_num):
    '''
    This function will return multiple properties for any given particle of any given type (dark, disk, or bulge)
    Inputs: filename is the name of the datafile
            particle_type is the type of particle (1 for dark, 2 for disk, and 3 for bulge)
            particle_num is the total number of particles
    Outputs: dist (units of kpc) is the magnitude of the distance of the particle
             vel (units of km/s) is the magnitude of the velocity of the particle
             particle_mass (units of solar mass) is the mass of the particle
    '''
    # assign variables to the outputs of the Read function
    time, N, data = Read(filename)

    # this variable allows the code to run for one specific particle type
    index = np.where(data['type'] == particle_type)

    # assign the mass of each particle to variable mass
    mass = data['m'][index]*(10 ** 10)

    # assign the data to different variables

    x = data['x'][index]
    y = data['y'][index]
    z = data['z'][index]
    
    vx = data['vx'][index]
    vy = data['vy'][index]
    vz = data['vz'][index]
    
    # determine how many particles there are of a certain type
    Npart = len(index[0])

    # initialize arrays for radius, velocity, and mass
    r = np.zeros((int(Npart),3))
    v = np.zeros((int(Npart),3))
    m = np.zeros((int(Npart),1))

    # loop through all particles of a given type
    for i in range(int(Npart)):
        
        # assign values to arrays
        r[i, 0] = x[i]
        r[i, 1] = y[i]
        r[i, 2] = z[i]
        v[i, 0] = vx[i]
        v[i, 1] = vy[i]
        v[i, 2] = vz[i]
        m[i] = mass[i]

    return r, v, m, Npart
