import argparse
import time

from astropy import units as u
import h5py
import numpy as np
import pyvista as pv
from spacepy.pybats import bats
from scipy.constants import m_p
import vtk

EARTH_DIPOLE_B0 = -31_000 * u.nT


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
    
    print('took', time.time() - start_time, 's')
    

def do_regrid(xbats, ybats, zbats, bx, by, bz, Ex, Ey, Ez, n, T, args):
    """Do regridding o unstructured grid to structured grid."""
    print('Regridding')

    # Get grid
    xaxis, yaxis, zaxis = get_new_grid(args)
    X, Y, Z = np.meshgrid(xaxis, yaxis, zaxis)

    # Make Polydata object
    point_cloud = pv.PolyData(np.transpose([xbats, ybats, zbats]))
    point_cloud['Bx'] = bx.flatten(order='F')
    point_cloud['By'] = by.flatten(order='F')
    point_cloud['Bz'] = bz.flatten(order='F')
    point_cloud['Ex'] = Ex.flatten(order='F')
    point_cloud['Ey'] = Ey.flatten(order='F')
    point_cloud['Ez'] = Ez.flatten(order='F')
    point_cloud['n'] = n.flatten(order='F')
    point_cloud['T'] = T.flatten(order='F')

    points_search = pv.PolyData(np.transpose([X.flatten(), Y.flatten(), Z.flatten()]))
    interp = vtk.vtkPointInterpolator()  
    interp.SetInputData(points_search)
    interp.SetSourceData(point_cloud)
    interp.GetKernel().SetRadius(args.interp_radius)
    interp.Update()

    interp_result = pv.PolyData(interp.GetOutput())

    # Pull out of Polydata object
    regrid_data = {}

    for var in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'n', 'T']:
        regrid_data[var] = interp_result[var].reshape(X.shape)
        
    regrid_data['xaxis'] = xaxis
    regrid_data['yaxis'] = yaxis
    regrid_data['zaxis'] = zaxis
    
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
    

def get_new_grid(args):
    """Definds the new grid to regrid to."""
    xaxis = np.arange(-3, 15, args.grid_spacing)
    yaxis = np.arange(-15, 15, args.grid_spacing)
    zaxis = np.arange(-5, 5, args.grid_spacing)

    return xaxis, yaxis, zaxis
    
    
def subtract_dipole(bx, by, bz, xbats, ybats, zbats):
    """Subtract dipole from the magnetic field values (store external field)"""
    rbats = np.sqrt(xbats**2 + ybats**2 + zbats**2)
    
    bx = bx - 3 * xbats.value * zbats.value * EARTH_DIPOLE_B0 / rbats.value**5
    by = by - 3 * ybats.value * zbats.value * EARTH_DIPOLE_B0 / rbats.value**5
    bz = bz - (3 * zbats.value**2 - rbats.value**2) * EARTH_DIPOLE_B0 / rbats.value**5

    return bx, by, bz    


if __name__ == '__main__':
    main()
