import gampixpy
from gampixpy import units, mobility

import os
import yaml
import numpy as np
import torch

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

        # calculate the span of the anode in TPC coordinates
        # based on the provided center
        
        # need to handle the case where the pitch doesn't evenly divide the span of the anode
        for volume_name, volume_dict in self['drift_volumes'].items():
            anode_center = torch.tensor([volume_dict['anode_center']['x'],
                                         volume_dict['anode_center']['y'],
                                         volume_dict['anode_center']['z']]).float()

            vertical_axis = torch.tensor([volume_dict['anode_vertical']['x'],
                                          volume_dict['anode_vertical']['y'],
                                          volume_dict['anode_vertical']['z']]).float()
            vertical_axis = vertical_axis/torch.sqrt(torch.inner(vertical_axis, vertical_axis))

            drift_axis = torch.tensor([volume_dict['drift_direction']['x'],
                                       volume_dict['drift_direction']['y'],
                                       volume_dict['drift_direction']['z']]).float()
            drift_axis = drift_axis/torch.sqrt(torch.inner(drift_axis, drift_axis))
            # drift_axis = drift_axis/np.linalg.norm(drift_axis)

            message = "anode vertical axis is not perpendicular to given drift axis!"
            assert torch.inner(vertical_axis, drift_axis) == 0, message

            horizontal_axis = torch.linalg.cross(vertical_axis,
                                                 drift_axis)

            half_span_horizontal = horizontal_axis*volume_dict['anode_span']['width']/2
            half_span_vertical = vertical_axis*volume_dict['anode_span']['height']/2

            anode_corners = [anode_center - half_span_horizontal - half_span_vertical,
                             anode_center - half_span_horizontal + half_span_vertical,
                             anode_center + half_span_horizontal - half_span_vertical,
                             anode_center + half_span_horizontal + half_span_vertical,
                             ]

            depth_span = drift_axis*volume_dict['depth']
            cathode_corners = [anode_center - half_span_horizontal - half_span_vertical + depth_span,
                               anode_center - half_span_horizontal + half_span_vertical + depth_span,
                               anode_center + half_span_horizontal - half_span_vertical + depth_span,
                               anode_center + half_span_horizontal + half_span_vertical + depth_span,
                               ]

            anode_mask = [1, 1, 1, 1, 0, 0, 0, 0]
            # hard-coded adjacency based on the definition of the corners above
            connectivity = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 0],
                                         [1, 0, 0, 1, 0, 1, 0, 0],
                                         [1, 0, 0, 1, 0, 0, 1, 0],
                                         [0, 1, 1, 0, 0, 0, 0, 1],
                                         [1, 0, 0, 0, 0, 1, 1, 0],
                                         [0, 1, 0, 0, 1, 0, 0, 1],
                                         [0, 0, 1, 0, 1, 0, 0, 1],
                                         [0, 0, 0, 1, 0, 1, 1, 0]], dtype = bool)

            volume_dict.update({'anode_center': anode_center,
                                'anode_vertical': vertical_axis,
                                'anode_horizontal': horizontal_axis,
                                'drift_axis': drift_axis,
                                'anode_corners': anode_corners,
                                'cathode_corners': cathode_corners,
                                'corners': torch.stack(anode_corners + cathode_corners),
                                'connectivity': connectivity,
                                })
        
        # print (self['drift_volumes'])
        # self['tile_volume_edges'] = (np.linspace(self['anode']['x_lower_bound'],
        #                                          self['anode']['x_upper_bound'],
        #                                          self['n_tiles_x']+1),
        #                              np.linspace(self['anode']['y_lower_bound'],
        #                                          self['anode']['y_upper_bound'],
        #                                          self['n_tiles_y']+1))
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

        # self['n_pixels_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['pixels']['pitch'])
        # self['n_pixels_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['pixels']['pitch'])

        # self['n_tiles_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['coarse_tiles']['pitch'])
        # self['n_tiles_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['coarse_tiles']['pitch'])

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

# far_detector_params = DetectorConfig(os.path.join(gampixpy.__path__[0],
#                                                       'detector_config',
#                                                       'far_detector.yaml'))
# far_readout_params = ReadoutConfig(os.path.join(gampixpy.__path__[0],
#                                                     'readout_config',
#                                                     'far_detector.yaml'))
