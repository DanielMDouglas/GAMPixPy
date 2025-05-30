import gampixpy
from gampixpy import units, mobility

import os
import yaml
import numpy as np

class Config (dict):
    """
    Config(config_filename)

    Initialize a new config dict from a yaml file.  This class serves
    as the parent class for specialized config classes for detector,
    physics, and readout parameter settings.  This class defines the
    methods for reading specifications from the input, resolving units
    to the internal unit scheme, and computing derived parameters,
    returning a dict-like object containing all parameters and their values.

    Parameters
    ----------
    config_filename : path_like
        A string or os.path-like object pointing to a yaml file containing
        definitions for parameters.

    Returns
    -------
    out : Config
        A dict-like object containing input and derived parameters.

    See Also
    --------
    DetectorConfig : Sub-class for parsing parameters for detector geometry
                     and steering.
    PhysicsConfig : Sub-class for parsing parameters for physics processes
                    (recombination, charge mobility, etc.)
    ReadoutConfig : Sub-class for parsing parameters for readout details.

    Examples
    --------
    >>> c = Config('path/to/config.yaml')

    """
    def __init__(self, config_filename):
        self.config_filename = config_filename

        self._parse_config()
        self._compute_derived_parameters()
        
    def _parse_config(self):
        # parse a config (yaml) file and store values
        # internally as a dict
        with open(self.config_filename) as config_file:
            self.update(yaml.load(config_file, Loader = yaml.FullLoader).items())

        # where quantities with units are specified,
        # resolve them to the internal unit system
        self.update(self._resolve_units(self))

    def _resolve_units(self, sub_dict):
        if 'value' in sub_dict and 'unit' in sub_dict:
            numerical_value = sub_dict['value']
            unit = units.unit_parser(sub_dict['unit'])
            resolved_dict = numerical_value*unit
        else:
            resolved_dict = {}
            for key, value in sub_dict.items():
                if type(value) == dict:
                    resolved_dict[key] = self._resolve_units(value)
                else:
                    resolved_dict[key] = value

        return resolved_dict

class DetectorConfig (Config):
    """
    DetectorConfig(config_filename)

    Initialize a new detector config dict from a yaml file.  This class
    reads specifications from the input, resolves units to the internal
    unit scheme, and computes derived parameters, returning a dict-like
    object containing all parameters and their values.

    Parameters
    ----------
    config_filename : path_like
        A string or os.path-like object pointing to a yaml file containing
        definitions for detector geometry parameters.

    Returns
    -------
    out : DetectorConfig
        A dict-like object containing input and derived parameters for
        detector geometry.

    See Also
    --------
    Config : Parent config class which does not computation of derived
             parameters.
    PhysicsConfig : Similar config class for parsing parameters for
                    physics processes (recombination, charge mobility,
                    etc.)
    ReadoutConfig : Similar config class for parsing parameters for
                    readout details.

    Examples
    --------
    >>> dc = DetectorConfig('path/to/config.yaml')

    """
    def _compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        # nothing to compute yet!
        # in the future, this should provide a method for coordinate
        # system transforms between input, internal, and output coords

        return

class PhysicsConfig (Config):
    """
    PhysicsConfig(config_filename)

    Initialize a new pysics config dict from a yaml file.  This class
    reads specifications from the input, resolves units to the internal
    unit scheme, and computes derived parameters, returning a dict-like
    object containing all parameters and their values.

    Parameters
    ----------
    config_filename : path_like
        A string or os.path-like object pointing to a yaml file containing
        definitions for physics parameters.

    Returns
    -------
    out : PhysicsConfig
        A dict-like object containing input and derived parameters for
        physics processes.

    See Also
    --------
    Config : Parent config class which does not computation of derived
             parameters.
    DetectorConfig : Similar config class for parsing parameters for
                     detector geometry and steering.
    ReadoutConfig : Similar config class for parsing parameters for
                    readout details.

    Examples
    --------
    >>> pc = PhysicsConfig('path/to/config.yaml')

    """
    def _compute_derived_parameters(self):
        mobility_model = mobility.MobilityModel(self)
        self['charge_drift'].update(mobility_model.compute_parameters())

        return

class ReadoutConfig (Config):
    """
    ReadoutConfig(config_filename)

    Initialize a new readout config dict from a yaml file.  This class
    reads specifications from the input, resolves units to the internal
    unit scheme, and computes derived parameters, returning a dict-like
    object containing all parameters and their values.

    Parameters
    ----------
    config_filename : path_like
        A string or os.path-like object pointing to a yaml file containing
        definitions for readout parameters.

    Returns
    -------
    out : ReadoutConfig
        A dict-like object containing input and derived parameters for
        readout electronics simulation.

    See Also
    --------
    Config : Parent config class which does not computation of derived
             parameters.
    DetectorConfig : Similar config class for parsing parameters for
                     detector geometry and steering.
    PhysicsConfig : Similar config class for parsing parameters for
                    physics processes (recombination, charge mobility,
                    etc.)

    Examples
    --------
    >>> rc = ReadoutConfig('path/to/config.yaml')

    """
    def _compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        # need to handle the case where the pitch doesn't evenly divide the span of the anode
        self['n_pixels_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['pixels']['pitch'])
        self['n_pixels_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['pixels']['pitch'])

        self['n_tiles_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['coarse_tiles']['pitch'])
        self['n_tiles_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['coarse_tiles']['pitch'])

        self['tile_volume_edges'] = (np.linspace(self['anode']['x_lower_bound'],
                                                 self['anode']['x_upper_bound'],
                                                 self['n_tiles_x']+1),
                                     np.linspace(self['anode']['y_lower_bound'],
                                                 self['anode']['y_upper_bound'],
                                                 self['n_tiles_y']+1))
        return

default_detector_params = DetectorConfig(os.path.join(gampixpy.__path__[0],
                                                      'detector_config',
                                                      'default.yaml'))
default_physics_params = PhysicsConfig(os.path.join(gampixpy.__path__[0],
                                                    'physics_config',
                                                    'default.yaml'))
default_readout_params = ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                    'readout_config',
                                                    'default.yaml'))

far_detector_params = DetectorConfig(os.path.join(gampixpy.__path__[0],
                                                      'detector_config',
                                                      'far_detector.yaml'))
far_readout_params = ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                    'readout_config',
                                                    'far_detector.yaml'))
