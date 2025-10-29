import h5py
import numpy as np

from gampixpy.readout_objects import coarse_tile_dtype, pixel_dtype
from gampixpy.input_parsing import meta_dtype

class OutputManager:
    """
    OutputManager(output_filename)

    Manager class for writing simulation output to a file on disk.  This
    class defines how to handle the HDF5 datasets, as well as how to add
    entries to a dataset on disk sequentially.

    Attributes
    ----------
    output_filename : string or os.path-like
        Path to output destination on disk.
    outfile : h5py.File
        Opened file handle for the output file.
    n_tracks : int
        Number of tracks written to the output file so far.
    
    """
    def __init__(self, output_filename):
        self.output_filename = output_filename

        self.outfile = h5py.File(output_filename, 'w')

        self.outfile.create_dataset('coarse_hits',
                                    shape = (0,),
                                    dtype = coarse_tile_dtype,
                                    maxshape = (None,))
        self.outfile.create_dataset('pixel_hits',
                                    shape = (0,),
                                    dtype = pixel_dtype,
                                    maxshape = (None,))
        self.outfile.create_dataset('meta',
                                    shape = (0,),
                                    dtype = meta_dtype,
                                    maxshape = (None,))
        self.n_tracks = 0 # track how many tracks have been written so far

    def add_file_metadata(self, readout_config, detector_config, physics_config):
        """
        om.add_file_metadata()

        Add file-level metadata to the output file.  This is to easily track
        configuration data and other relevant environment details so that
        the simulation can be reproduced with no other input (besides the
        original G4-level input).

        """

        self.outfile.attrs[]

    def add_entry(self, track, meta, event_id = None):
        """
        om.add_entry(track, meta, event_id = None)

        Add a new row to the output datasets

        Parameters
        ----------
        track : Track object
            A track which has been fully simulated and has coarse tile and
            pixel hits populated.
        meta : array-like[float]
            Metadata array for this entry.  Metadata dtype is described in
            input_parsing.meta_dtype.
        event_id : int
            Index to assign this image in the output dataset.  If not
            specified, use the current value of the internal counter.
        
        """
        self._add_track(track, event_id)
        self._add_meta(meta, event_id)

        self.n_tracks += 1
        
    def _add_track(self, track, event_id = None):
        coarse_tile_sample_array, pixel_sample_array = track.to_array()

        if event_id:
            coarse_tile_sample_array[:]['event id'] = event_id
            pixel_sample_array[:]['event id'] = event_id
        else:
            coarse_tile_sample_array[:]['event id'] = self.n_tracks
            pixel_sample_array[:]['event id'] = self.n_tracks
        
        n_coarse_hits = coarse_tile_sample_array.shape[0]
        n_coarse_hits_prev = self.outfile['coarse_hits'].shape[0]

        n_pixel_hits = pixel_sample_array.shape[0]
        n_pixel_hits_prev = self.outfile['pixel_hits'].shape[0]

        self.outfile['coarse_hits'].resize((n_coarse_hits+n_coarse_hits_prev,))
        self.outfile['coarse_hits'][n_coarse_hits_prev:] = coarse_tile_sample_array

        self.outfile['pixel_hits'].resize((n_pixel_hits+n_pixel_hits_prev,))
        self.outfile['pixel_hits'][n_pixel_hits_prev:] = pixel_sample_array

    def _add_meta(self, meta, event_id):
        if event_id:
            meta[:]['event id'] = event_id
        else:
            meta[:]['event id'] = self.n_tracks
        
        self.outfile['meta'].resize((self.n_tracks + 1,))
        self.outfile['meta'][self.n_tracks:] = meta
