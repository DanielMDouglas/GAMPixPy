from gampixpy.readout_objects import NULL_EVENT, NULL_LABEL
from gampixpy.input_parsing import EdepSimParser
from gampixpy.config import default_config_manager, ConfigManager

import h5py
import numpy as np
import pickle

class OutputParser:
    """
    OutputParser

    Initialize a new OutputParser analysis object.  This class provides
    a nicer interface for paging through simulated detector output.

    Parameters
    ----------
    gampix_sim_output : file-like
        A string or os.path-like object pointing to an hdf5 file containing
        some simulated detector output.

    Returns
    -------
    out : OutputParser
        An OutputParser object
    
    See Also
    --------
    CrossReferenceParser : Class for paging through input and outputs of GAMPixPy
                           using labels from truth-tracking mode.

    Examples
    --------
    >>> op = OutputParser('path/to/gampixpy_output.hdf5')
    >>> op.event_id = 0
    >>> op.label = 3
    >>> pix, tile, meta = op.get_data()

    This is equivalent to 
    >>> op = OutputParser('path/to/gampixpy_output.hdf5')
    >>> pix, tile, meta = op.get_data(0, 3)

    """
    def __init__(self, gampix_sim_output):
        self.gampix_sim = gampix_sim_output

        self._file_handle = h5py.File(self.gampix_sim)

        self._event_id = NULL_EVENT
        self._label = NULL_LABEL

        self._pixel_hit_event_mask = np.zeros_like(self._file_handle['pixels']['event id'],
                                                   dtype = bool)
        self._tile_hit_event_mask = np.zeros_like(self._file_handle['tiles']['event id'],
                                                    dtype = bool)
        self._meta_mask = np.zeros_like(self._file_handle['meta']['event id'],
                                        dtype = bool)

        self._pixel_hit_label_mask = np.empty(0, dtype = bool)
        self._tile_hit_label_mask = np.empty(0, dtype = bool)

        self._event_indices = np.unique(self._file_handle['meta']['event id'])
        self._label_list = np.empty(0)

    @property
    def event_id(self):
        """
        Current event index
        """
        return self._event_id

    @event_id.setter
    def event_id(self, event_id):
        # check if event_id is valid in the context of the file
        self._event_id = event_id

        self.eval_event_mask()
        self.eval_label_mask()

    @property
    def label(self):
        """
        Current label selection
        """
        return self._label

    @label.setter
    def label(self, label):
        # check if this is a valid label for the current event?
        # assert label in self._label_list
        
        self._label = label

        self.eval_label_mask()

    @property
    def label_list(self):
        """
        Get the list of available labels for the currently specified event.
        """
        return self._label_list

    def eval_event_mask(self):
        self._pixel_hit_event_mask = self._file_handle['pixels']['event id'] == self._event_id
        self._tile_hit_event_mask = self._file_handle['tiles']['event id'] == self._event_id
        self._meta_mask = self._file_handle['meta']['event id'] == self._event_id

    def eval_label_mask(self):
        event_pix = self._file_handle['pixels'][self._pixel_hit_event_mask]
        event_pix_label = event_pix['label']
        event_pix_attr = event_pix['attribution']
        event_pix_wf = event_pix['waveform']
        
        event_pix_ind = np.argmax(np.sum(event_pix_attr*event_pix_wf[:,:,None],
                                         axis = 1),
                                  axis = 1)
        maj_label_pix = np.array([pix_label[pix_ind]
                                  for pix_label, pix_ind
                                  in zip(event_pix_label,
                                         event_pix_ind)])
        
        event_tile = self._file_handle['tiles'][self._tile_hit_event_mask]
        event_tile_label = event_tile['label']
        event_tile_attr = event_tile['attribution']
        event_tile_wf = event_tile['waveform']

        event_tile_ind = np.argmax(np.sum(event_tile_attr*event_tile_wf[:,:,None],
                                         axis = 1),
                                  axis = 1)
        maj_label_tile = np.array([tile_label[tile_ind]
                                  for tile_label, tile_ind
                                  in zip(event_tile_label,
                                         event_tile_ind)])

        self._label_list = np.unique(maj_label_pix)
        
        # need to handle label reduction
        self._pixel_hit_label_mask = maj_label_pix == self._label
        self._tile_hit_label_mask = maj_label_tile == self._label
        
    def get_configs(self):
        detector_config = pickle.loads(self._file_handle.attrs['detector config'])
        physics_config = pickle.loads(self._file_handle.attrs['physics config'])
        readout_config = pickle.loads(self._file_handle.attrs['readout config'])

        config_manager = ConfigManager(detector_config = detector_config,
                                       physics_config = physics_config,
                                       readout_config = readout_config)

        return config_manager
        
    def get_data(self,
                 event_id = None,
                 label = None,
                 label_reduction_method = 'max'):
        if event_id is not None:
            self.event_id = event_id
        if label is not None:
            self.label = label
        
        sel_pixel_hits = self._file_handle['pixels'][self._pixel_hit_event_mask][self._pixel_hit_label_mask]
        sel_tile_hits = self._file_handle['tiles'][self._tile_hit_event_mask][self._tile_hit_label_mask]
        sel_meta = self._file_handle['meta'][self._meta_mask]

        return sel_pixel_hits, sel_tile_hits, sel_meta

    def __iter__(self, *args, **kwargs):
        # iterate through all outputs
        for event_id in self._event_indices:
            self.event_id = event_id
            for label in self._label_list:
                self.label = label
                yield self.get_data()

class CrossReferenceParser:
    """
    CrossReferenceParser

    Initialize a new CrossReferenceParser analysis object.  This class
    provides a clean interface for iterating through events and labels
    in simulation output and the corresponding inputs.

    Right now, only edep-sim hdf5 input files are supported.

    Parameters
    ----------
    input_edepsim : path-like
        A string or os.path-like object pointing to an hdf5 file containing
        edep-sim segment data.
    gampix_sim_output : path-like
        A string or os.path-like object pointing to an hdf5 file containing
        some simulated detector output.
    physics_config : PhysicsConfig object
        Physics configuration.  Some early physics processes are handled
        by the input parser (for now!), such as charge/light yield
        calculation.
    readout_config : ReadoutConfig object
        Config object containing specifications for readout planes.

    See Also
    --------
    OutputParser : Class for paging through simulated detector output.
    EdepSimParser : Class for paging through input edepsim segment data.
    
    """
    def __init__(self,
                 input_edepsim,
                 gampix_sim_output,
                 ):
        
        self._event_id = NULL_EVENT
        self._label = NULL_LABEL

        self._label_list = np.empty(0)

        self.gampixsim = gampix_sim_output
        self.output_parser = OutputParser(gampix_sim_output)

        self._config_manager = self.get_configs()
        self._event_indices = self.output_parser._event_indices

        self.edepsim = input_edepsim
        self.input_parser = EdepSimParser(input_edepsim,
                                          config_manager = self._config_manager,
                                          )

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, event_id):
        self._event_id = event_id
        self.output_parser.event_id = event_id
        self._label_list = self.output_parser._label_list               

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def label_list(self):
        """
        Get the list of available labels for the currently specified event.
        """
        return self._label_list
        
    def get_configs(self):
        config_manager = self.output_parser.get_configs()

        return config_manager

    def get_data(self,
                 *args,
                 event_id = None,
                 label = None,
                 **kwargs):
        if event_id:
            self.event_id = event_id
        if label:
            self.label = label

        edepsim_event = self.input_parser.get_segments(self.event_id)
        label_mask = edepsim_event[6] == self.label
        
        label_edepsim_event = (edepsim_event[0][label_mask],
                               edepsim_event[1][label_mask],
                               edepsim_event[2][label_mask],
                               edepsim_event[3][label_mask],
                               edepsim_event[4][label_mask],
                               edepsim_event[5][label_mask],
                               )

        gampixpy_event = self.output_parser.get_data(self.event_id,
                                                     self.label,
                                                     **kwargs)

        return label_edepsim_event, gampixpy_event

    def __iter__(self, *args, **kwargs):
        # iterate through all outputs
        for event_id in self._event_indices:
            self.event_id = event_id
            for label in self._label_list:
                self.label = label
                yield self.get_data()

class SparseTensorConverter (OutputParser):
    """
    SparseTensorConverter

    Initialize a new converter object which can produce MinkowskiEngine
    SparseTensors
    
    Parameters
    ----------
    gampix_sim_output : file-like
        A string or os.path-like object pointing to an hdf5 file containing
        some simulated detector output.

    Returns
    -------
    out : SparseTensorConverter
        An SparseTensorConverter object which can read hdf5 output and produce
        sparsetensors with an iterable interface (as in a torch dataloader).
    
    See Also
    --------
    OutputParser : Parent class for paging through outputs of GAMPixPy.
    CrossReferenceParser : Class for paging through input and outputs of GAMPixPy
                           using labels from truth-tracking mode.

    """

    def get_sparse_tensor(self,
                          event_id = None,
                          label = None,
                          label_reduction_method = 'max'):

        import MinkowskiEngine as ME

        data = self.get_data(event_id,
                             label,
                             label_reduction_method)

        pixel, tile, meta = data

        pixel_coords = torch.tensor([])
        pixel_features = torch.tenor([])

        tile_coords = torch.tensor([])
        tile_features = torch.tensor([])

        # pixel_st =
        # tile_st = 
    
    def __iter__(self, *args, **kwargs):
        for event_id in self._event_indices:
            self.event_id = event_id
            for label in self._label_list:
                self.label = label
                yield self.get_sparse_tensor()
