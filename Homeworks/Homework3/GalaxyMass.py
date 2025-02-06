# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:10:10 2025

@author: savan
"""

import numpy as np
import astropy.units as u
from ReadFile import Read
from ParticleProperties import ParticleProperties
import pandas as pd

def ComponentMass(filename, particle_type):
    '''
        This function computes the total mass of any desired galaxy component
        Inputs: filename is the name of the file
                particle_type is the type of the particle (1 for halo, 2 for disk, and 3 for bulge)
        Outputs: total_mass is the total mass of the galaxy component in units of 10^12 solar mass
    '''
    # assigning variables to the outputs of the Read function
    time, N, data = Read(filename)

    # this variable allows the code to run for one specific particle type
    index = np.where(data['type'] == particle_type)

    # initializing the count of how many particles are of the particle type
    number_of_type = 0
    # looping through all N particles in the data set
    for i in range(int(N)):
        # if the particle type is equal to the desired type, one is added to the total count
        if data['type'][i] == particle_type:
            number_of_type += 1

    # assigning m from the Read function to be used in this function
    # the [index] part ensures that the data being assigned to the variable is for the right particle type
    # units will be in solar mass but having units was disrupting the code written in later cells
    mass = data['m'][index]*(10 ** 10)

    # initializing the total mass of the galaxy component
    total_mass = 0
    # looping through every particle with the proper particle type
    for i in range(number_of_type - 1):
        # adding the mass of that particle to the total mass
        total_mass += mass[i]

    return total_mass
    

# assigning variables to the outputs of ComponentMass for each galaxy and for each particle type
# using np.round to round each value to 3 decimal places

MW_halo = np.round(ComponentMass("MW_000.txt", 1), 3)
MW_disk = np.round(ComponentMass("MW_000.txt", 2), 3)
MW_bulge = np.round(ComponentMass("MW_000.txt", 3), 3)

M31_halo = np.round(ComponentMass("M31_000.txt", 1), 3)
M31_disk = np.round(ComponentMass("M31_000.txt", 2), 3)
M31_bulge = np.round(ComponentMass("M31_000.txt", 3), 3)

M33_halo = np.round(ComponentMass("M33_000.txt", 1), 3)
M33_disk = np.round(ComponentMass("M33_000.txt", 2), 3)
M33_bulge = np.round(ComponentMass("M33_000.txt", 3), 3)


# printing the values with a short description before the value in units of 10^12 solar mass:

print(f"MW halo (dark matter) mass: {MW_halo/1e12:.3f}e+12")
print(f"MW disk mass: {MW_disk/1e12:.3f}e+12")
print(f"MW bulge mass: {MW_bulge/1e12:.3f}e+12")

print(f"M31 halo (dark matter) mass: {M31_halo/1e12:.3f}e+12")
print(f"M31 disk mass: {M31_disk/1e12:.3f}e+12")
print(f"M31 bulge mass: {M31_bulge/1e12:.3f}e+12")

print(f"M33 halo (dark matter) mass: {M33_halo/1e12:.3f}e+12")
print(f"M33 disk mass: {M33_disk/1e12:.3f}e+12")
print(f"M33 bulge mass: {M33_bulge/1e12:.3f}")


# computing the total mass for each galaxy by adding each galaxy component mass
MW_total = MW_halo + MW_disk + MW_bulge
M31_total = M31_halo + M31_disk + M31_bulge
M33_total = M33_halo + M33_disk + M33_bulge

# computing values for the local group by adding each component of the separate galaxies
LG_halo = MW_halo + M31_halo + M33_halo
LG_disk = MW_disk + M31_disk + M33_disk
LG_bulge = MW_bulge + M31_bulge + M33_bulge
LG_total = LG_halo + LG_disk + LG_bulge

# computing the fbar for each galaxy
# fbar = (total stellar mass / total mass) where total mass includes the mass from dark matter
MW_fbar = np.round(((MW_disk + MW_bulge) / (MW_total)), 3)
M31_fbar = np.round(((M31_disk + M31_bulge) / (M31_total)), 3)
M33_fbar = np.round(((M33_disk + M33_bulge) / (M33_total)), 3)
LG_fbar = np.round(((LG_disk + LG_bulge) / (LG_total)), 3)


# organizing data to be used in a table
data = {
    "Galaxy Name": ["MW", "M31", "M33", "local group"],
    "Halo Mass (M_☉)": [f'{MW_halo/1e12:.3f}e+12', f'{M31_halo/1e12:.3f}e+12', f'{M33_halo/1e12:.3f}e+12', f'{LG_halo/1e12:.3f}e+12'],
    "Disk Mass (M_☉)": [f'{MW_disk/1e12:.3f}e+12', f'{M31_disk/1e12:.3f}e+12', f'{M33_disk/1e12:.3f}e+12', f'{LG_disk/1e12:.3f}e+12'],
    "Bulge Mass (M_☉)": [f'{MW_bulge/1e12:.3f}e+12', f'{M31_bulge/1e12:.3f}e+12', f'{M33_bulge/1e12:.3f}e+12', f'{LG_bulge/1e12:.3f}e+12'],
    "Total Mass (M_☉)": [f'{MW_total/1e12:.3f}e+12', f'{M31_total/1e12:.3f}e+12', f'{M33_total/1e12:.3f}e+12', f'{LG_total/1e12:.3f}e+12'],
    "fbar": [MW_fbar, M31_fbar, M33_fbar, LG_fbar]
}

# using pandas to produce a table with all the data
df = pd.DataFrame(data)
print(df)

# calculating and printing certain values to answer homework questions:

print(f'MW total mass: {MW_total:.3e}')
print(f'M31 total mass: {M31_total:.3e}')
# calculating the mass difference between MW and M31
mass_difference = M31_total - MW_total
print(f'the difference in mass between the MW and M31 is {mass_difference:.3e}')

# calculating the stellar mass (SM) of MW and M31 by adding the disk mass and bulge mass
MW_SM = MW_disk + MW_bulge
M31_SM = M31_disk + M31_bulge
print(f'MW stellar mass: {MW_SM:.3e}')
print(f'M31 stellar mass: {M31_SM:.3e}')
# calculating the different in stellar mass between MW and M31
SM_difference = M31_SM - MW_SM
print(f'the difference in stellar mass between the MW and M31 is {SM_difference:.3e}')

# calculating the dark matter ratio between MW and M31
DM_ratio = M31_halo / MW_halo
print(f'the ratio of dark matter in M31 to dark matter in the MW is {DM_ratio:.3}')

print(f'MW baryon fraction: {MW_fbar:.3}')
print(f'M31 baryon fraction: {M31_fbar:.3}')
print(f'M33 baryon fraction: {M33_fbar:.3}')
