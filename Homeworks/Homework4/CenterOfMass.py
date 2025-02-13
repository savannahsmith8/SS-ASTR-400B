# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:40:48 2025

@author: savan
"""

# Homework 4
# Center of Mass Position and Velocity
# Savannah Smith

# import modules
import numpy as np
import astropy.units as u
import astropy.table as tbl

from ReadFile import Read

class CenterOfMass:
# class to define COM position and velocity properties of a given galaxy and simulation snapshot

    def __init__(self, filename, ptype):
        ''' class to calculate the 6-D phase-space position of a galaxy's center of mass using
        a specified particle type. 
            
            PARAMETERS
            ----------
            filename : `str`
                snapshot file
            ptype : `int; 1, 2, or 3`
                particle type to use for COM calculations
        '''
     
        # reading data in the given file using Read
        self.time, self.total, self.data = Read(filename)                                                                                             

        # creating an array to store indexes of particles of desired Ptype                                
        self.index = np.where(self.data['type'] == ptype)

        # storing the mass, positions, and velocities of only the particles of the given type
        self.m = self.data['m'][self.index]
        self.x = self.data['x'][self.index]
        self.y = self.data['y'][self.index]
        self.z = self.data['z'][self.index]
        self.vx = self.data['vx'][self.index]
        self.vy = self.data['vy'][self.index]
        self.vz = self.data['vz'][self.index]
        

    def COMdefine(self,a,b,c,m):
        ''' method to compute the COM of a generic vector quantity by direct weighted averaging.
        
        PARAMETERS
        ----------
        a : `float or np.ndarray of floats`
            first vector component
        b : `float or np.ndarray of floats`
            second vector component
        c : `float or np.ndarray of floats`
            third vector component
        m : `float or np.ndarray of floats`
            particle masses
        
        RETURNS
        -------
        a_com : `float`
            first component on the COM vector
        b_com : `float`
            second component on the COM vector
        c_com : `float`
            third component on the COM vector
        '''
        
        # generic equation used: x_COM = sum(xi*mi)/sum(mi)
        # xcomponent center of mass
        a_com = np.sum(a * m) / np.sum(m)
        # ycomponent center of mass
        b_com = np.sum(b * m) / np.sum(m)
        # zcomponent center of mass
        c_com = np.sum(c * m) / np.sum(m)

        return a_com, b_com, c_com
       
        
    def COM_P(self, delta):
        '''Method to compute the position of the center of mass of the galaxy 
        using the shrinking-sphere method.

        PARAMETERS
        ----------
        delta : `float, optional`
            error tolerance in kpc. Default is 0.1 kpc
        
        RETURNS
        ----------
        p_COM : `np.ndarray of astropy.Quantity'
            3-D position of the center of mass in kpc
        '''                                                                     

        # determining the Center of Mass Position                                                                                    

        # step 1:
        
        # first guess at the COM position by calling COMdefine                                                   
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        # computing the magnitude of the COM position vector
        r_COM = np.sqrt( (x_COM ** 2) + (y_COM ** 2) + (x_COM ** 2) )

        # iterative process to determine the center of mass                                                            

        # changing reference frame to COM frame                                                                          
        # computing the difference between particle coordinates and the first guess at COM position
        x_new = self.x - x_COM
        y_new = self.y - y_COM
        z_new = self.z - z_COM
        r_new = np.sqrt( (x_new ** 2) + (y_new ** 2) + (z_new ** 2) )
        
        # finding the max 3D distance of all particles from the guessed COM                                               
        # will re-start at half that radius (reduced radius)                                                           
        r_max = np.max(r_new)/2.0

        # step 2:

        # picking an initial value for the change in COM position between the first guess above and the new one computed from half that volume
        # should be larger than the input tolerance (delta) initially
        change = 1000.0   # in kpc

        # starting the iterative process to determine the center of mass position                                                 
        # delta is the tolerance for the difference between the old and new COM    
        
        while (change > delta):
            
            # selecting all particles within the reduced radius (starting from original x,y,z,m)
            index2 = np.where(r_new < r_max)
            x2 = self.x[index2]
            y2 = self.y[index2]
            z2 = self.z[index2]
            m2 = self.m[index2]

            # refined COM position:                                                                                    
            # computing the center of mass position using the particles in the reduced radius
            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2, y2, z2, m2)
            # computing the new 3D COM position
            r_COM2 = np.sqrt( (x_COM2 ** 2) + (y_COM2 ** 2) + (z_COM2 ** 2) )

            # determining the difference between the previous center of mass position and the new one.                                                                                         
            change = np.abs(r_COM - r_COM2)                                                                                              
            # print ("CHANGE = ", change)   

            # before loop continues, resetting : r_max, particle separations and COM 
            
            # reduce the volume by a factor of 2 again                                                                 
            r_max /= 2.0                                                                                            
            # print ("maxR", r_max)    
            
            # setting the center of mass positions to the refined values                                                   
            x_COM = x_COM2
            y_COM = y_COM2
            z_COM = z_COM2
            r_COM = r_COM2

            # create an array (np.array) to store the COM position                                                                                                                                                       
            p_COM = np.array([x_COM, y_COM, z_COM])

        # step 3:

        # setting the units to kpc using astropy
        # rounding all values to 2 decimal places
        # returning the COM position vector
        p_COM = np.round(p_COM, 2)*u.kpc

        return p_COM
    
            
    def COM_V(self, x_COM, y_COM, z_COM):
        ''' Method to compute the center of mass velocity based on the center of mass
        position.

        PARAMETERS
        ----------
        x_COM : 'astropy quantity'
            The x component of the center of mass in kpc
        y_COM : 'astropy quantity'
            The y component of the center of mass in kpc
        z_COM : 'astropy quantity'
            The z component of the center of mass in kpc
            
        RETURNS
        -------
        v_COM : `np.ndarray of astropy.Quantity'
            3-D velocity of the center of mass in km/s
        '''
        
        # the max distance from the center that is used to determine the center of mass velocity                   
        rv_max = 15.0*u.kpc

        # determining the position of all particles relative to the center of mass position (x_COM, y_COM, z_COM)
        xV = self.x - x_COM.value
        yV = self.y - y_COM.value
        zV = self.z - z_COM.value
        rV = np.sqrt( (xV ** 2) + (yV ** 2) + (zV ** 2) )
        
        # determining the index for those particles within the max radius
        indexV = np.where(rV < rv_max.value)
        
        # determining the velocity and mass of those particles within the mass radius
        
        # Note that x_COM, y_COM, z_COM are astropy quantities and you can only subtract one astropy quantity from another
        # So, when determining the relative positions, assign the appropriate units to self.x
        vx_new = self.vx[indexV]
        vy_new = self.vy[indexV]
        vz_new = self.vz[indexV]
        m_new =  self.m[indexV]
        
        # computing the center of mass velocity using those particles
        vx_COM, vy_COM, vz_COM = self.COMdefine(vx_new, vy_new, vz_new, m_new)
        
        # creating an array to store the COM velocity
        v_COM = np.array([vx_COM, vy_COM, vz_COM])

        # returning the COM vector
        # setting the units to km/s using astropy
        # rounding all values to 2 decimal places
        v_COM = np.round(v_COM, 2)*u.km/u.s
    
        return v_COM

# creating a Center of mass object for the MW, M31 and M33
# at Snapshot 0 using Disk Particles only:

MW_COM = CenterOfMass("MW_000.txt", 2)
M31_COM = CenterOfMass("M31_000.txt", 2)
M33_COM = CenterOfMass("M33_000.txt", 2)

# homework questions (section 6):
    
    
# question 1:
# finding the COM position (in kpc) and velocity (in km/s) vector for the MW, M31, and M33

# COM positions for MW, M31, and M33:

MW_COM_p = MW_COM.COM_P(0.1)
print(f"The COM position for MW is {MW_COM_p}")

M31_COM_p = M31_COM.COM_P(0.1)
print(f"The COM position for M31 is {M31_COM_p}")

M33_COM_p = M33_COM.COM_P(0.1)
print(f"The COM position for M33 is {M33_COM_p}")

print("")

# COM velocities for MW, M31, and M33:

MW_COM_v = MW_COM.COM_V(MW_COM_p[0], MW_COM_p[1], MW_COM_p[2])
print(f"The COM velocity for MW is {MW_COM_v}")

M31_COM_v = M31_COM.COM_V(M31_COM_p[0], M31_COM_p[1], M31_COM_p[2])
print(f"The COM velocity for M31 is {M31_COM_v}")

M33_COM_v = M33_COM.COM_V(M33_COM_p[0], M33_COM_p[1], M33_COM_p[2])
print(f"The COM velocity for M33 is {M33_COM_v}")


# question 2:
# finding the magnitude of the current separation (in kpc) and velocity (in km/s) between the MW and M31

# computing the distance between MW and M31 for x, y, and z
MW_M31_pos_diff = MW_COM_p - M31_COM_p
# calculating the separation using the position differences
MW_M31_sep = np.sqrt( (MW_M31_pos_diff[0] ** 2) + (MW_M31_pos_diff[1] ** 2) + (MW_M31_pos_diff[2] ** 2))
print(f"The separation between MW and M31 is {np.round(MW_M31_sep, 2)}")

# computing the velocity differences between MW and M31 for vx, vy, and vz
MW_M31_vel_diff = MW_COM_v - M31_COM_v
# calculating the overall velocity difference
MW_M31_vel = np.sqrt( (MW_M31_vel_diff[0] ** 2) + (MW_M31_vel_diff[1] ** 2) + (MW_M31_vel_diff[2] ** 2) )
print(f"The velocity between MW and M31 is {np.round(MW_M31_vel, 2)}")


# question 3:
# finding the magnitude of the current separation (in kpc) and velocity (in km/s) between M33 and M31

# computing the distance between M31 and M33 for x, y, and z
M31_M33_pos_diff = M31_COM_p - M33_COM_p
# calculating the separation using the position differences
M31_M33_sep = np.sqrt( (M31_M33_pos_diff[0] ** 2) + (M31_M33_pos_diff[1] ** 2) + (M31_M33_pos_diff[2] ** 2))
print(f"The separation between M31 and M33 is {np.round(M31_M33_sep, 2)}")

# computing the velocity differences between M31 and M33 for vx, vy, and vz
M31_M33_vel_diff = M31_COM_v - M33_COM_v
# calculating the overall velocity difference
M31_M33_vel = np.sqrt( (M31_M33_vel_diff[0] ** 2) + (M31_M33_vel_diff[1] ** 2) + (M31_M33_vel_diff[2] ** 2) )
print(f"The velocity between M31 and M33 is {np.round(M31_M33_vel, 2)}")


# question 4:
# given that M31 and the MW are about to merge, why is the iterative process to determine the COM so important?

# The iterative process is essential for determining the positions and velocities of the MW and M31 
# because their kinematics directly affect each other. Because the two galaxies are about to merge, 
# they are relatively close to each other on a universal scale so they are constantly being influenced
# by the other's gravitational forces and chaotic events. By using an iterative process, we are able
# to continuously update the positions and velocities of each galaxy based on how their mass distributions
# change throughout their motions (using the first function, COMdefine). As the galaxies get closer to
# merging, their interactions will become stronger and stronger, causing their centers of mass to change
# continuously.