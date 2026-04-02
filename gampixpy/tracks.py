import numpy as np
import torch

from gampixpy.readout_objects import dtype_factory, NULL_LABEL
from gampixpy import config

class Track:
    """
    Track (sample_position, sample_time, sample_charge)

    General-purpose data class for storing representations of an
    ionization event in the detector.

    Attributes
    ----------

    raw_track : dict
        Dict containing sample position vectors (key: 'position'),
        sample ionization time (key 'time'), sample charge values
        (key 'charge'), and sample labels (key 'label').  All are
        array-like.
    drifted_track : dict
        Dict containing sample position 3-vectors ('position'), sample
        arrival times ('time') and charge after attenuation ('charge').
    pixel_samples : list[CoarseGridSample]
        List of coarse tile hits found by detector simulation.
    coarse_tiles_samples : list[PixelSample]
        List of pixel hits found by detector simulation.

    See Also
    --------
    PixelSample : Data class for pixel hits.
    CoarseTileSample : Data class for tile hits.
    
    """
    def __init__(self,
                 sample_position,
                 sample_time,
                 sample_charge,
                 sample_labels,
                 ):
        self.raw_track = {'position': sample_position,
                          'time': sample_time,
                          'charge': sample_charge,
                          'label': sample_labels}

        self.tpc_track = {}
        self.drifted_track = {}

        self.pixel_samples = []
        self.coarse_tiles_samples = []

    def to_array(self, readout_config = config.default_readout_params):
        """
        track.to_array()

        Generate a numpy array with the simulated hit data contained
        in this object for saving to a summary HDF5 file.

        Parameters
        ----------
        readout_config : ReadoutConfig object
            Config object containing specifications for tile and pixel size, gaps,
            threshold, noise, etc.

        Returns
        -------
        coarse_tile_sample_array : Flattened numpy.array of coarse hit
            data with dtype described in readout_objects.coarse_tile_dtype.
        pixel_sample_array : Flattened numpy.array of pixel hit data with
            dtype described in readout_objects.coarse_tile_dtype.

        """
        tile_dtype, pixel_dtype = dtype_factory(readout_config)

        coarse_tile_sample_array = np.array([(0,
                                              tile_record.tile_tpc,
                                              tile_record.tile_pos[0],
                                              tile_record.tile_pos[1],
                                              tile_record.trigger_depth,
                                              tile_record.trigger_time,
                                              tile_record.waveform,
                                              tile_record.attribution,
                                              tile_record.labels)
                                             for tile_record in self.coarse_tiles_samples],
                                            dtype = tile_dtype)
        pixel_sample_array = np.array([(0,
                                        pixel_record.pixel_tpc,
                                        pixel_record.pixel_pos[0],
                                        pixel_record.pixel_pos[1],
                                        pixel_record.trigger_depth[0],
                                        pixel_record.trigger_time,
                                        pixel_record.waveform,
                                        pixel_record.attribution,
                                        pixel_record.labels)
                                       for pixel_record in self.pixel_samples],
                                      dtype = pixel_dtype)

        return coarse_tile_sample_array, pixel_sample_array

null_track = Track(torch.tensor([]),
                   torch.tensor([]),
                   torch.tensor([]),
                   torch.tensor([]),
                   )
