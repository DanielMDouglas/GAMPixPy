from gampixpy.readout_objects import NULL_EVENT, NULL_LABEL
from gampixpy.input_parsing import EdepSimParser
from gampixpy.config import default_config_manager, ConfigManager
from gampixpy.coordinates import CoordinateManager

import h5py
import numpy as np
import pickle
import torch
        
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
        self._config_manager = self.get_configs()

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
        if event_id is not None:
            self.event_id = event_id
        if label is not None:
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
    def __init__(self, gampix_sim_output, batch_size = 1):
        import spconv.pytorch as spconv

        self.gampix_sim = gampix_sim_output

        self._file_handle = h5py.File(self.gampix_sim)
        self._config_manager = self.get_configs()

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

        self._batch_size = batch_size

        # sample order randomization
        self._sample_load_order = np.empty(0,)
        self._sequential = False
        self.gen_sample_load_order()

    @property
    def sample_load_order(self):
        return self._sample_load_order

    @sample_load_order.setter
    def sample_load_order(self, sample_load_order):
        self._sample_load_order = sample_load_order

    def gen_sample_load_order(self):
        # set the order that events/images within a given file
        # are sampled
        # This should be redone after each file is loaded
        n_events = len(np.unique(self._file_handle['meta']['event id']))
        if self._sequential:
            self.sample_load_order = np.arange(n_events)
        else:
            self.sample_load_order = np.random.choice(n_events,
                                                      size = n_events,
                                                      replace = False)

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

    def get_features_and_indices(self,
                                 event_id = None,
                                 label = None,
                                 label_reduction_method = 'max',
                                 batch_index = 0,
                                 **kwargs):

        data = self.get_data(event_id,
                             label,
                             label_reduction_method)
        pixels, tiles, meta = data

        rc = self._config_manager.readout_config

        wf_len = rc['coarse_tiles']['integration_length']
        pix_pitch = rc['pixels']['pitch']
        clock_int = rc['coarse_tiles']['clock_interval']
     
        batch_size = 1
     
        x = torch.tensor(pixels['pixel x']).repeat_interleave(wf_len)
        y = torch.tensor(pixels['pixel y']).repeat_interleave(wf_len)
        t = (torch.tensor(pixels['trig t'])[:,None] + clock_int*torch.arange(wf_len)[None,:]).flatten()
        q = torch.tensor(pixels['waveform']).flatten()

        features = torch.stack([x, y, t, q]).T
        
        x_ind = (x/pix_pitch).int()
        x_ind -= torch.min(x_ind)
     
        y_ind = (y/pix_pitch).int()
        y_ind -= torch.min(y_ind)
     
        t_ind = (t/clock_int).int()
        t_ind -= torch.min(t_ind)
     
        batch_ind = batch_index*torch.ones_like(x_ind)
        
        indices = torch.stack([batch_ind, x_ind, y_ind, t_ind]).T

        return features, indices

    def get_sparsetensor(self,
                         *args,
                         **kwargs):

        features, indices = self.get_features_and_indices(*args, **kwargs)
        
        spatial_shape = [torch.max(indices[:,1])+1,
                         torch.max(indices[:,2])+1,
                         torch.max(indices[:,3])+1,
                         ]

        pixel_st = spconv.SparseConvTensor(features,
                                           indices,
                                           spatial_shape,
                                           1)
     
        return pixel_st

    def get_vertex_depth(self, event_id = None):
        """
        Calculate depth of an event based on the vertex
        stored in the metadata.  This may produce unexpected
        results when the vertex is not well-defined.
        """        
        if event_id is not None:
            self.event_id = event_id

        # get the metadata for this event 
        event_meta = self._file_handle['meta'][self._meta_mask]
        # get the vertex position
        source_point_exp = torch.tensor(np.array([event_meta['vertex x'],
                                                  event_meta['vertex y'],
                                                  event_meta['vertex z'],
                                                  ])).T

        # rotate to tpc coordinate system
        cm = CoordinateManager(self._config_manager)
        source_point_tpc = cm.to_tpc_coords(source_point_exp)[0]

        depth = source_point_tpc[3]
        return depth
    
    def __iter__(self, *args, **kwargs):
        """
        Iterator interface to produce SparseTensor batches.
        Batch size is defined upon construction of the convertor object.
        """
        
        im_count = 0
        
        for event_id in self._sample_load_order:
            self.event_id = event_id
            for label in self._label_list:
                self.label = label
                depth = self.get_vertex_depth()
                feats, inds = self.get_features_and_indices(batch_index = im_count)

                if im_count == 0:
                    batch_depth = depth[None]
                    batch_feats = feats
                    batch_inds = inds
                else:
                    batch_depth = torch.cat((batch_depth, depth[None]))
                    batch_feats = torch.cat((batch_feats, feats))
                    batch_inds = torch.cat((batch_inds, inds))

                im_count += 1

                if im_count == self._batch_size:
                    spatial_shape = [torch.max(batch_inds[:,1])+1,
                                     torch.max(batch_inds[:,2])+1,
                                     torch.max(batch_inds[:,3])+1,
                                     ]
                    st = spconv.SparseConvTensor(batch_feats,
                                                 batch_inds,
                                                 spatial_shape,
                                                 self._batch_size)

                    yield st, batch_depth

                    im_count = 0
