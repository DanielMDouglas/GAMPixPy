from gampixpy.readout_objects import NULL_EVENT, NULL_LABEL
from gampixpy.input_parsing import EdepSimParser
from gampixpy.config import default_physics_params, default_readout_params

import h5py
import numpy as np

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
    >>> pix, coarse, meta = op.get_data()

    This is equivalent to 
    >>> op = OutputParser('path/to/gampixpy_output.hdf5')
    >>> pix, coarse, meta = op.get_data(0, 3)

    """
    def __init__(self, gampix_sim_output):
        self.gampix_sim = gampix_sim_output

        self._file_handle = h5py.File(self.gampix_sim)

        self._event_id = NULL_EVENT
        self._label = NULL_LABEL

        self._pixel_hit_event_mask = np.zeros_like(self._file_handle['pixel_hits']['event id'],
                                                   dtype = bool)
        self._coarse_hit_event_mask = np.zeros_like(self._file_handle['coarse_hits']['event id'],
                                                    dtype = bool)
        self._meta_mask = np.zeros_like(self._file_handle['meta']['event id'],
                                        dtype = bool)

        self._pixel_hit_label_mask = np.empty(0, dtype = bool)
        self._pixel_coarse_label_mask = np.empty(0, dtype = bool)

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
        self._pixel_hit_event_mask = self._file_handle['pixel_hits']['event id'] == self.event_id
        self._coarse_hit_event_mask = self._file_handle['coarse_hits']['event id'] == self.event_id
        self._meta_mask = self._file_handle['meta']['event id'] == self.event_id

    def eval_label_mask(self):
        event_pix = self._file_handle['pixel_hits'][self._pixel_hit_event_mask]
        event_pix_label = event_pix['label']
        event_pix_attr = event_pix['attribution']
        maj_label_pix = np.array([pix_label[np.argmax(pix_attr)]
                                  for pix_label, pix_attr in zip(event_pix_label,
                                                                 event_pix_attr)])

        event_coarse = self._file_handle['coarse_hits'][self._coarse_hit_event_mask]
        event_coarse_label = event_coarse['label']
        event_coarse_attr = event_coarse['attribution']
        maj_label_coarse = np.array([coarse_label[np.argmax(coarse_attr)]
                                     for coarse_label, coarse_attr in zip(event_coarse_label,
                                                                          event_coarse_attr)])

        self._label_list = np.unique(maj_label_pix)
        
        # need to handle label reduction
        self._pixel_hit_label_mask = maj_label_pix == self.label
        self._coarse_hit_label_mask = maj_label_coarse == self.label
        # self._meta_mask *= self._file_handle['meta']['label'] == self.label

    def get_data(self,
                 event_id = None,
                 label = None,
                 label_reduction_method = 'max'):
        if event_id:
            self.event_id = event_id
        if label:
            self.label = label

        sel_pixel_hits = self._file_handle['pixel_hits'][self._pixel_hit_event_mask][self._pixel_hit_label_mask]
        sel_coarse_hits = self._file_handle['coarse_hits'][self._coarse_hit_event_mask][self._coarse_hit_label_mask]
        sel_meta = self._file_handle['meta'][self._meta_mask]

        return sel_pixel_hits, sel_coarse_hits, sel_meta

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
                 physics_config = default_physics_params,
                 readout_config = default_readout_params,
                 ):
        self.edepsim = input_edepsim
        self.input_parser = EdepSimParser(input_edepsim,
                                          physics_config = physics_config,
                                          readout_config = readout_config,
                                          )
        
        self.gampixsim = gampix_sim_output
        self.output_parser = OutputParser(gampix_sim_output)

        self._event_id = NULL_EVENT
        self._label = NULL_LABEL

        self._physics_config = physics_config
        self._readout_config = readout_config

        self._event_indices = self.output_parser._event_indices
        self._label_list = np.empty(0)

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
