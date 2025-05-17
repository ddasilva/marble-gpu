import argparse
import time
from datetime import timedelta

from astropy import units as u
from cupyx.scipy.spatial import KDTree
import pandas as pd
import h5py
import numpy as np
import pyvista as pv
from spacepy.pybats import bats
from scipy.constants import m_p
import vtk
from cupyx.scipy.spatial import KDTree
import cupy as cp
from tqdm import tqdm

EARTH_DIPOLE_B0 = -31_000 * u.nT
R_INNER = 2.5
MAX_NEIGHBOR_DISTANCE = 25


def main():
    start_time = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--interp-radius', default=0.15, type=float)
    parser.add_argument('--grid-spacing', default=0.15, type=float)
    args = parser.parse_args()

    # Load SWMF file
    print(f'Loading file {args.input}')
    mhd = bats.IdlFile(args.input)

    print('Found variables:')
    print(mhd.keys())

    # Load coordinates
    print('Loading data')
    xbats = mhd['x'] * u.R_earth
    ybats = mhd['y'] * u.R_earth
    zbats = mhd['z'] * u.R_earth    

    # Load number density
    n = mhd['rho'] / m_p / 1e27
    n *= u.cm**(-3)

    # Load Magnetic Field
    bx = mhd['bx'] * u.nT
    by = mhd['by'] * u.nT
    bz = mhd['bz'] * u.nT

    # Load Flow Velocity    
    ux = mhd['ux'] * u.km/u.s
    uy = mhd['uy'] * u.km/u.s
    uz = mhd['uz'] * u.km/u.s
    
    # Load ElectricFfield
    Ex, Ey, Ez = -np.cross([ux, uy, uz], [bx, by, bz], axis=0)

    units = bx.unit * ux.unit
    Ex *= units
    Ey *= units
    Ez *= units

    better_units = u.mV/u.m
    Ex = Ex.to(better_units)
    Ey = Ey.to(better_units)
    Ez = Ez.to(better_units)    

    # Load Pressure and Temperature
    p = mhd['p'] * u.nPa
    T = (p / n).to(u.eV)
    
    # Subtract dipole values from Magnetic Field
    print('Subtracting dipole')
    bx, by, bz = subtract_dipole(bx, by, bz, xbats, ybats, zbats)

    # Do regrid
    regrid_data = do_regrid(xbats, ybats, zbats, bx, by, bz, Ex, Ey, Ez, n, T, args)

    # Write output
    write_regrid_data(args, regrid_data)

    
def do_regrid(xbats, ybats, zbats, bx, by, bz, Ex, Ey, Ez, n, T, args):
    """Do regridding on unstructured grid to uniform grid."""
    print('Regridding')
    xaxis, yaxis, zaxis, taxis, radii = get_new_grid(xbats, ybats, zbats, args)
    X, Y, Z = np.meshgrid(xaxis, yaxis, zaxis, indexing='ij')

    print('Building KDTree')
    tree = KDTree(cp.array([xbats.value, ybats.value, zbats.value]).T)
    print('Done')

    print("Querying")
    query_points = cp.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    k = 16
    d, I = tree.query(query_points, k=k)
    print('Done')

    print('Interpolating')
    vars = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'n', 'T']
    target_shape = X.shape + (2,)
    regrid_data = {}
    for var in vars:
        regrid_data[var] = np.zeros(target_shape)
        
    scale = d.mean(axis=1)
    scale_ = cp.zeros((scale.size, k))
    for i in range(k):
        scale_[:, i] = scale

    # Use Gaussian RBFs with scale apprximated by average neighbor distances
    weights = np.exp(-(d/scale_)**2)
    norm = weights.sum(axis=1)
    
    for i in range(k):
        weights[:, i] /= norm

    regrid_data['Bx'][:, :, :, 0] = cp.sum(cp.array(bx)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['By'][:, :, :, 0] = cp.sum(cp.array(by)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['Bz'][:, :, :, 0] = cp.sum(cp.array(bz)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['Ex'][:, :, :, 0] = cp.sum(cp.array(Ex)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['Ey'][:, :, :, 0] = cp.sum(cp.array(Ey)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['Ez'][:, :, :, 0] = cp.sum(cp.array(Ez)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['n'][:, :, :, 0] = cp.sum(cp.array(n)[I] * weights, axis=1).get().reshape(X.shape)
    regrid_data['T'][:, :, :, 0] = cp.sum(cp.array(T)[I] * weights, axis=1).get().reshape(X.shape)

    for var in vars:
        regrid_data[var][:, :, :, 1] = regrid_data[var][:, :, :, 0]

    # for var in vars:
    #     mask = (d.min(axis=1) > MAX_NEIGHBOR_DISTANCE).reshape(X.shape).get()
    #     regrid_data[var][mask] = np.nan
        
    print('Done')
    
    regrid_data['xaxis'] = xaxis
    regrid_data['yaxis'] = yaxis
    regrid_data['zaxis'] = zaxis
    regrid_data['taxis'] = taxis
    regrid_data['r_inner'] = R_INNER
    
    return regrid_data


def write_regrid_data(args, regrid_data):
    """Write regridded data to HDF5 file"""
    print(f'Writing to {args.output}')
    
    hdf_file = h5py.File(args.output, 'w')

    for key, value in regrid_data.items():
        hdf_file[key] = value

    hdf_file['Bx'].attrs['units'] = 'nT'
    hdf_file['By'].attrs['units'] = 'nT'
    hdf_file['Bz'].attrs['units'] = 'nT'

    hdf_file['Ex'].attrs['units'] = 'mV/m'
    hdf_file['Ey'].attrs['units'] = 'mV/m'
    hdf_file['Ez'].attrs['units'] = 'mV/m'

    hdf_file['n'].attrs['units'] = 'cm^-3'
    hdf_file['T'].attrs['units'] = 'eV'

    hdf_file.close()
    

def get_new_grid(xbats, ybats, zbats, args):
    """Definds the new grid to regrid to."""
    # xaxis = np.arange(-15, 15, args.grid_spacing)
    # yaxis = np.arange(-15, 15, args.grid_spacing)
    # zaxis = np.arange(-15, 15, args.grid_spacing)

    # Too much
    #xaxis = np.arange(xbats.min().value, xbats.max().value, args.grid_spacing)
    #yaxis = np.arange(ybats.min().value, ybats.max().value, args.grid_spacing)
    #zaxis = np.arange(zbats.min().value, zbats.max().value, args.grid_spacing)    

    # Use OpenGGCM Grid
    dfs = {}

    for dim in 'xyz':
        dfs[dim] = pd.read_csv(
            f'data/OpenGGCM_grids/overwiew_7M_now_11.8Mcells/grid{dim}.txt',
            sep='\\s+',
            names=[dim, 'delta', 'unused2'],
            skiprows=1
        )

    xaxis = -dfs['x'].x[::-1]
    yaxis = dfs['y'].y
    zaxis = dfs['z'].z
    time_window = timedelta(days=5).total_seconds()
    taxis = np.array([-time_window, time_window])
    
    radii = 2.5 * np.sqrt(dfs['x'].delta**2 +
                          dfs['y'].delta**2 +
                          dfs['z'].delta**2)

    print('batsx', xbats.min(), xbats.max())
    print('ggscmx', xaxis.min(), xaxis.max())

    m = (xaxis > xbats.min()) & (xaxis < xbats.max())
    radii = radii[m]
    xaxis = xaxis[m]
    yaxis = yaxis[(yaxis > ybats.min()) & (yaxis < ybats.max())]
    zaxis = zaxis[(zaxis > zbats.min()) & (zaxis < zbats.max())]
    
    return xaxis, yaxis, zaxis, taxis, radii
    
    
def subtract_dipole(bx, by, bz, xbats, ybats, zbats):
    """Subtract dipole from the magnetic field values (store external field)"""
    rbats = np.sqrt(xbats**2 + ybats**2 + zbats**2)
    
    bx = bx - 3 * xbats.value * zbats.value * EARTH_DIPOLE_B0 / rbats.value**5
    by = by - 3 * ybats.value * zbats.value * EARTH_DIPOLE_B0 / rbats.value**5
    bz = bz - (3 * zbats.value**2 - rbats.value**2) * EARTH_DIPOLE_B0 / rbats.value**5

    return bx, by, bz    


if __name__ == '__main__':
    main()
