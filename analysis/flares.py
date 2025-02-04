import h5py
import numpy as np
import cmasher as cmr
from unyt import Msun, yr, g, s, Mpc, erg, c, G, mp, sigma_thompson

# flares specific quantities
h = 0.677
radius = 14./h * Mpc
volume = (4./3) * np.pi * (radius)**3 
_volume = volume.to('Mpc**3').value
radiative_efficiency = 0.1

# units
luminosity_units = erg / s
mass_units = Msun
accretion_rate_units = Msun / yr

# the new master file
data_dir = '../../data'
data_file = f'{data_dir}/flares_blackholes.hdf5'

# get list of tag and sims

with h5py.File(data_file, 'r') as hf:

    sims = list(hf.keys())
    tags = list(hf[sims[0]].keys())
    print(sims)
    print(tags)
    hf[f'{sims[0]}/{tags[0]}'].visit(print)

snapshots = tags
snapshot_redshifts = np.array([10, 9, 8, 7, 6, 5])
redshifts = snapshot_redshifts

redshift_colours = cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))

# grab weights
x, y, z, deltas, sigmas, weights = np.loadtxt('../weights_grid.txt', skiprows=1, unpack = True, usecols = (2,3,4,6,7,8), delimiter=',')

# the conversion rate from the ionising photon luminosity (Q) to the Halpha luminosity
# this is calculated in the theory notebook
ionising_to_Halpha_conversion = 3.066E-12 * erg


def calculate_bolometric_luminosity(
        accretion_rate,
        radiative_efficiency=radiative_efficiency):

    """
    Calculate bolometric luminosity from accretion rate
    """

    return radiative_efficiency * accretion_rate * c**2


def calculate_eddington_accretion_rate(
        blackhole_mass,
        radiative_efficiency=radiative_efficiency):

    return ((4 * np.pi * G * blackhole_mass * mp)
            / (radiative_efficiency * sigma_thompson * c))


def get_quantities(tag, group='master'):

    """
    Function to get quantities based on the master file.
    """

    with h5py.File(data_file, 'r') as hf:

        quantities = {}

        quantities['attenuated_stellar_halpha_ew'] = np.array([])
        quantities['attenuated_stellar_halpha_luminosity'] = np.array([])
        quantities['intrinsic_stellar_halpha_ew'] = np.array([])
        quantities['intrinsic_stellar_halpha_luminosity'] = np.array([])

        quantities['stellar_masses'] = np.array([])
        quantities['stellar_bolometric_luminosities'] = np.array([])
        quantities['blackhole_masses'] = np.array([])
        quantities['simulation'] = []
        quantities['accretion_rates'] = {k: np.array([]) for k in hf[f'{sims[0]}/{tag}/Galaxy/{group}/blackhole_accretion_rate'].keys()}
        quantities['weights'] = np.array([])

        for sim, weight in zip(sims, weights):
            
            mdata = hf[f'{sim}/{tag}/Galaxy/master']
            data = hf[f'{sim}/{tag}/Galaxy/{group}']
            n = len(data['blackhole_mass'][:])
            quantities['weights'] = np.concatenate((quantities['weights'], np.ones(n) * weight))
        
            quantities['simulation'] += [sim] * n

            # stellar quantities    
            for k in ['attenuated_stellar_halpha_ew', 'attenuated_stellar_halpha_luminosity', 'intrinsic_stellar_halpha_ew', 'intrinsic_stellar_halpha_luminosity']:
                quantities[k] = np.concatenate((quantities[k], mdata[k][:]))

            quantities['stellar_masses'] = np.concatenate((quantities['stellar_masses'], mdata['stellar_mass'][:]))
            quantities['stellar_bolometric_luminosities'] = np.concatenate((quantities['stellar_bolometric_luminosities'], mdata['stellar_bolometric_luminosity2'][:]))

            # blackhole properties
            quantities['blackhole_masses'] = np.concatenate((quantities['blackhole_masses'], data['blackhole_mass'][:]))
            for k in data['blackhole_accretion_rate'].keys():
                quantities['accretion_rates'][k] = np.concatenate((quantities['accretion_rates'][k], data[f'blackhole_accretion_rate/{k}'][:]))

    quantities['simulation'] = np.array(quantities['simulation'])

    quantities['stellar_masses'] *= 1E10 * mass_units
    quantities['blackhole_masses'] *= mass_units
    quantities['stellar_bolometric_luminosities'] *= luminosity_units

    for k in quantities['accretion_rates'].keys():
        quantities['accretion_rates'][k] *= accretion_rate_units

    # calculate Eddington accretion rates
    quantities['eddington_accretion_rates'] = calculate_eddington_accretion_rate(quantities['blackhole_masses'])

    # calculate Eddington ratios
    quantities['eddington_ratios'] = {}
    for k in quantities['accretion_rates'].keys():
        quantities['eddington_ratios'][k] = quantities['accretion_rates'][k] / quantities['eddington_accretion_rates']

    # calculate bolometric_luminosity
    quantities['bolometric_luminosities'] = {}
    for k in quantities['accretion_rates'].keys():
        quantities['bolometric_luminosities'][k] = calculate_bolometric_luminosity(quantities['accretion_rates'][k])

    return quantities

