import numpy as np

from gampixpy.readout_objects import coarse_tile_dtype, pixel_dtype

class Track:
    """
    Track (sample_position, sample_time, sample_charge)

    General-purpose data class for storing representations of an
    ionization event in the detector.

    Attributes
    ----------

    raw_track : dict
        Dict containing sample position vectors (key: 'position'),
        sample ionization time (key 'time'), and sample charge 
        values (key 'charge').  All are array-like.
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
    def __init__(self, sample_position, sample_time, sample_charge):
        self.raw_track = {'position': sample_position,
                          'time': sample_time,
                          'charge': sample_charge}

        self.tpc_track = {}
        self.drifted_track = {}

        self.pixel_samples = []
        self.coarse_tiles_samples = []

    def to_array(self):
        """
        track.to_array()

        Generate a numpy array with the simulated hit data contained
        in this object for saving to a summary HDF5 file.

        Parameters
        ----------
        None

        Returns
        -------
        coarse_tile_sample_array : Flattened numpy.array of coarse hit
            data with dtype described in readout_objects.coarse_tile_dtype.
        pixel_sample_array : Flattened numpy.array of pixel hit data with
            dtype described in readout_objects.coarse_tile_dtype.

        """
        coarse_tile_sample_array = np.array([(0,
                                              hit.coarse_cell_pos[0],
                                              hit.coarse_cell_pos[1],
                                              hit.coarse_measurement_depth,
                                              hit.coarse_measurement_time,
                                              hit.coarse_cell_measurement)
                                             for hit in self.coarse_tiles_samples],
                                            dtype = coarse_tile_dtype)
        pixel_sample_array = np.array([(0,
                                        hit.pixel_pos[0],
                                        hit.pixel_pos[1],
                                        hit.hit_depth,
                                        hit.hit_timestamp,
                                        hit.hit_measurement)
                                       for hit in self.pixel_samples],
                                      dtype = pixel_dtype)

        return coarse_tile_sample_array, pixel_sample_array
