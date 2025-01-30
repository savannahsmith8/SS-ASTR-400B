# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:36:07 2025

@author: savan
"""

import numpy as np
import astropy.units as u

def Read(datafile):
    '''
    This function will read a data file and collect the time, total number of particles, and various information about each particle
    Inputs: the name of a datafile as a text file
            This code is specific to handle the data contained within MW_000.txt and how that text file is organized
    Outputs: time (units of Myr) is the time specified by the text file
             N is the total number of particles specified by the text file
             type is the type of particle with type 1 representing dark matter, type 2 representing disk stars, and type 3 representing bulge stars
             mass (units of 10^10 solar masses) and represents the mass of the particle
             x, y, z (units of kpc) are the positions of the particle measured from the center of mass position of the Milky Way (MW)
             vx, vy, vz (units of km/s) are the velocities of the particle using a cartesian coordinate system centered on the location of the MW
    '''
    file = open(datafile, 'r')

    # collecting time information:
    # reading the first line of the datafile
    line1 = file.readline()
    # assigning variables to the two groups of characters in the first line
    label, value = line1.split()
    # assigning the value variable to the time variable and assigning units of Myr
    time = float(value)*u.Myr

    # collecting number of particles information:
    # reading the second line of the datafile
    line2 = file.readline()
    # assigning variables to the two groups of characters in the second line
    label, value = line2.split()
    # assigning the value variable to the N variable for the total number of particles
    N = float(value)

    file.close()

    # collecting data information in a format that translates to arrays:
    
    # np.genfromtxt: allows the code to process column header information
    # dtype = None: allows for the column's data types to be interpreted
    # names = True: allows for the array names to match the labels
    # skip_header = 3: skips the first 3 lines (which display the time, N, and units)
    data = np.genfromtxt(datafile, dtype = None, names = True, skip_header = 3)

    return time, N, data
