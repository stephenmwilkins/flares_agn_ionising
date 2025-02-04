import numpy as np
import flares_utility.analyse as analyse
from unyt import (
    c,
    dimensionless,
    Msun,
    Mpc,
    yr,
    # Lsun,
    g,
    s,
    G,
    mp,
    erg,
    sigma_thompson)



# EAGLE specific quantities
# little-h
h = 0.677

radius = 14./h * Mpc
volume = (4./3) * np.pi * (radius)**3 


# radiative efficiency
radiative_efficiency = 0.1

# accretion rate units (already corrected for h)
blackhole_mass_units = Msun

# master-file units
blackhole_mass_units = 1E10 * Msun
blackhole_accretion_rate_units = 6.446E23 * g / s  # / h
print(blackhole_accretion_rate_units)

luminosity_units = erg / s

# accretion rate units (not corrected for h previously)
accretion_rate_units = Msun / yr 
seed_mass = 1E5 * Msun / h

# # These are the units we want to return
# accretion_rate_units = 'Msun/yr'
# mass_units = 'Msun'
# luminosity_units = 

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


# def load_data(
#         filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
#         tag='010_z005p000',
#         mass_limit=6.5):

#     """
#     Loads data for all galaxies in all simulations.
#     """

    
#     flares = analyse.analyse(filename, default_tags=False)

#     quantities = []
    
#     quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
#                   'name': 'Mstar', 'log10': True})
#     quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
#                     'name': 'BH_Mass', 'log10': True})
#     quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
#                     'name': 'BH_Mdot', 'log10': True})


#     D = flares.get_datasets(tag, quantities)
#     s = D['log10BH_Mass'] > mass_limit
 
#     blackhole_accretion_rate = D['BH_Mdot'][s] * accretion_rate_units
#     blackhole_mass = D['BH_Mass'][s] * blackhole_mass_units 
#     weights = D['weight'][s]

#     eddington_accretion_rate = calculate_eddington_accretion_rate(blackhole_mass)
#     bolometric_luminosity = calculate_bolometric_luminosity(blackhole_accretion_rate)

#     eddington_ratio = blackhole_accretion_rate/eddington_accretion_rate

#     return blackhole_mass, blackhole_accretion_rate, bolometric_luminosity, eddington_ratio, weights




def load_dataset_by_sim(
        dataset_id=['Galaxy', 'BH_Mass'],
        units=dimensionless,
        filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
        tag='010_z005p000',
        ):
    """
    Loads a single dataset
    """

    flares = analyse.analyse(filename, default_tags=False)

    dataset = flares.load_dataset(tag, *dataset_id)

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):
        dataset[sim] = np.array(dataset[sim]) * units

    return dataset


def load_flares(filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'):
    flares = analyse.analyse(filename, default_tags=False)
    return flares


def load_all_dataset(
        dataset_id=['Galaxy', 'BH_Mass'],
        units=dimensionless,
        filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
        tag='010_z005p000',
        ):
    """
    Loads a single dataset
    """

    flares = analyse.analyse(filename, default_tags=False)

    dataset_ = load_dataset_by_sim(
        dataset_id=dataset_id,
        units=units,
        filename=filename,
        tag=tag,
    )

    dataset = np.array([]) * units
    weights = np.array([])

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):
        weights = np.append(weights, w * np.ones(len(dataset_[sim])))
        dataset = np.append(dataset, dataset_[sim])

    return dataset, weights


def load_blackhole_data_by_sim(
        filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
        tag='010_z005p000',
        mass_limit=7.0):
    
    """
    Loads data for each individual simulation
    """

    flares = analyse.analyse(filename, default_tags=False)

    blackhole_masses = flares.load_dataset(tag, *['Galaxy', 'BH_Mass'])
    blackhole_accretion_rates = flares.load_dataset(tag, *['Galaxy', 'BH_Mdot'])

    bolometric_luminosities = {}

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):
        
        blackhole_masses[sim] = np.array(blackhole_masses[sim]) * blackhole_mass_units
        blackhole_accretion_rates[sim] = np.array(blackhole_accretion_rates[sim]) * blackhole_accretion_rate_units

        bolometric_luminosities[sim] = calculate_bolometric_luminosity(blackhole_accretion_rates[sim])
        bolometric_luminosities[sim] = bolometric_luminosities[sim].to('erg/s')

    return blackhole_masses, blackhole_accretion_rates, bolometric_luminosities


def load_all_blackhole_data(
        filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
        tag='010_z005p000',
        mass_limit=7.0):

    """
    Loads data for all galaxies in all simulations.
    """

    flares = analyse.analyse(filename, default_tags=False)

    blackhole_masses_, blackhole_accretion_rates_, bolometric_luminosities_ = load_blackhole_data_by_sim(
        filename=filename,
        tag=tag,
        mass_limit=mass_limit,
    )

    blackhole_masses = np.array([]) * blackhole_mass_units
    blackhole_accretion_rates = np.array([]) * blackhole_accretion_rate_units
    bolometric_luminosities = np.array([]) * luminosity_units
    weights = np.array([])

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):
        weights = np.append(weights, w * np.ones(len(blackhole_masses_[sim])))
        blackhole_masses = np.append(blackhole_masses, blackhole_masses_[sim])
        blackhole_accretion_rates = np.append(blackhole_accretion_rates, blackhole_accretion_rates_[sim])
        bolometric_luminosities = np.append(bolometric_luminosities, bolometric_luminosities_[sim])

    eddington_accretion_rates = calculate_eddington_accretion_rate(blackhole_masses)
    bolometric_luminosities = calculate_bolometric_luminosity(blackhole_accretion_rates)

    eddington_ratios = blackhole_accretion_rates/eddington_accretion_rates

    return blackhole_masses, blackhole_accretion_rates, bolometric_luminosities, eddington_ratios, weights