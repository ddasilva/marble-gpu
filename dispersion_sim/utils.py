
from dataclasses import dataclass

import ai.cs
from astropy import constants, units as u
import numpy as np
from scipy.constants import elementary_charge

import disco
import config


@dataclass
class DispersionSimSetup:
    particle_long_axis: np.ndarray
    particle_lat_axis: np.ndarray
    particle_energy_axis: np.ndarray
    particle_state: disco.ParticleState

def setup_dispersion_sim():
    particle_long_axis = np.array([0])
    particle_invlat_range = np.deg2rad([config.inv_lat_min, config.inv_lat_max])
    particle_lat_range = np.arccos(np.sqrt((1 / config.particle_height) * np.cos(particle_invlat_range)**2))
    particle_lat_axis = np.arange(*particle_lat_range, np.deg2rad(config.lat_step))

    particle_energy_axis = 10**np.linspace(np.log10(config.energy_min), np.log10(config.energy_max), config.energy_count) * u.eV

    # Build particle 
    particle_lat, particle_long, particle_energy = np.meshgrid(particle_lat_axis, particle_long_axis, particle_energy_axis, indexing='ij')
    pos_x, pos_y, pos_z = ai.cs.sp2cart(config.particle_height, particle_lat, particle_long )
    pos_x *= u.R_earth
    pos_y *= u.R_earth
    pos_z *= u.R_earth
    particle_vel = np.sqrt(2 * particle_energy / constants.m_p)
    ppar =  constants.m_p * particle_vel
    magnetic_moment = np.zeros(ppar.shape) * u.MeV/u.nT
    charge = elementary_charge * u.C

    particle_state = disco.ParticleState(pos_x.flatten(), pos_y.flatten(), pos_z.flatten(),
                                     ppar.flatten(), magnetic_moment.flatten(), constants.m_p, charge)

    return DispersionSimSetup(particle_long_axis, particle_lat_axis, particle_energy_axis, particle_state)