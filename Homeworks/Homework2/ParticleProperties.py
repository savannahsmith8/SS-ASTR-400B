# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:37:49 2025

@author: savan
"""

import numpy as np
import astropy.units as u
from ReadFile import Read

def ParticleProperties(filename, particle_type, particle_num):
    '''
    This function will return multiple properties for any given particle of any given type (dark, disk, or bulge)
    Inputs: filename is the name of the datafile
            type is the type of particle (1 for dark, 2 for disk, and 3 for bulge)
            N is the total number of particles
    Outputs: d_mag (units of kpc) is the magnitude of the distance of the particle
             v_mag (units of km/s) is the magnitude of the velocity of the particle
             mass (units of solar mass) is the mass of the particle
    '''
    # assigning variables to the outputs of the Read function
    time, N, data = Read(filename)

    # this variable allows the code to run for one specific particle type
    index = np.where(data['type'] == particle_type)

    # assigning all variables from the Read function to be used in this function
    # units added to variables using astropy
    # the [index] part ensures that the data being assigned to the variable is for the right particle type
    
    mass = data['m'][index]*(10 ** 10)*u.M_sun
    # assigning the mass of the individual particle to the variable particle_mass
    particle_mass = mass[particle_num-1]
    
    x = data['x'][index]*u.kpc
    y = data['y'][index]*u.kpc
    z = data['z'][index]*u.kpc
    
    vx = data['vx'][index]*u.km/u.s
    vy = data['vy'][index]*u.km/u.s
    vz = data['vz'][index]*u.km/u.s

    # calculating the distance of the particle in kpc
    # the index of N-1 will access the x, y, and z positions for the Nth particle
    d_mag = (( (x[particle_num-1] ** 2) + (y[particle_num-1] ** 2) + (z[particle_num-1] ** 2) ) ** (1/2))
    # rounding to 3 decimal places
    dist = np.around(d_mag, 3)

    # calculating the velocity of the particle in km/s
    # the index of N-1 will access the vx, vy, and vz velocities for the Nth particle
    v_mag = (( (vx[particle_num-1] ** 2) + (vy[particle_num-1] ** 2) + (vz[particle_num-1] ** 2) ) ** (1/2))
    # rounding to 3 decimal places
    vel = np.around(v_mag, 3)

    return dist, vel, particle_mass

# assigning variables to the outputs of the ParticleProperties function
# reading in text file MW_000.txt, analyzing type 2 particles (disk stars), and focusing on the 100th particle
distance, velocity, particle_mass = ParticleProperties("MW_000.txt", 2, 100)

print(f"the distance of the particle is {distance}")
print(f"the velocity of the particle is {velocity}")
print(f"the mass of the particle is {particle_mass}")