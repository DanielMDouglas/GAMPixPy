from gampixpy.config import default_detector_params, default_physics_params, default_readout_params
from gampixpy.readout_objects import PixelSample, CoarseGridSample
from gampixpy.coordinates import CoordinateManager

import numpy as np
import torch
import torchist

class ReadoutModel:
    """
    ReadoutModel(readout_config)

    Parent class for readout models.  This class defines the process of
    taking drifted point clouds and forming current series for each coarse
    tile and pixel in the volume (for some defined tile/pixel response model).

    Attributes
    ----------
    readout_config : ReadoutConfig object
        Config object containing specifications for tile and pixel size, gaps,
        threshold, noise, etc.
    physics_config : PhysicsConfig object
        Config object containing physics parameters for liquid Argon.
    detector_config : DetectorConfig object
        Config object containing specifications for TPC volume position and
        orientation.
    coordinate_manager : CoordinateManager object
        Coordinate manager defined by the provided physics_config.  This is a
        helper for transforming to and from TPC-specific coordinate systems.
    clock_start_time : float
        timestamp of the earliest arriving charge bundle.  This serves as the
        first clock tick of the event readout.
    coarse_tile_hits : array-like[CoarseTileHit]
        Array of coarse tile hits populated by tile_hit_finding method
    fine_pixel_hits : array-like[CoarseTileHit]
        Array of pixel hits populated by pixel_hit_finding method

    Notes
    -----
    tile_hit_finding and pixel_hit_finding methods are implemented by sub-classes.

    See Also
    --------
    GAMPixModel : Sub-class implementing a GAMPix-like hit-finding scheme.
    LArPixModel : Sub-class implementing a LArPix-like hit-finding scheme.    

    """
    def __init__(self,
                 readout_config = default_readout_params,
                 physics_config = default_physics_params,
                 detector_config = default_detector_params,
                 truth_tracking = True):
        self.readout_config = readout_config
        self.physics_config = physics_config
        self.detector_config = detector_config

        self.coordinate_manager = CoordinateManager(detector_config)

        self.truth_tracking = truth_tracking

        self.clock_start_time = 0

    def electronics_simulation(self, track, verbose = True, **kwargs):
        """
        readout.electronics_simulation(track,
                                       verbose = True,
                                       **kwargs)

        Steering function for the individual steps of electronics simulation.

        Parameters
        ----------
        track : Track object
            Input event representation containing a populated drifted_track.
        verbose : bool
            Print update messages as simulation progresses through each stage.
        
        """
        try:
            self.clock_start_time = torch.min(track.drifted_track['time'])
        except RuntimeError:
            print ("Encountered empty event!")
            message = """This is most often because the event lies totally outside of the specified detector volume"""
            print (message)
            print ("skipping...")

        if verbose:
            print ("simulating coarse grid...")
        # do coarse grid binning/time series formation
        coarse_tile_timeseries = self.tile_current_builder(track, **kwargs)
        if verbose:
            print ("coarse time series built")
            print ("coarse hit finding...")
        # find hits on coarse grid
        self.coarse_tile_hits = self.tile_hit_finding(track, coarse_tile_timeseries, **kwargs)

        if verbose:
            print ("simulating fine grid...")
        fine_pixel_timeseries = self.pixel_current_builder(track, self.coarse_tile_hits, **kwargs)
        if verbose:
            print ("pixel time series built")
            print ("pixel hit finding...")
        # find hits on fine pixels
        self.fine_pixel_hits = self.pixel_hit_finding(track, fine_pixel_timeseries, **kwargs)

        if verbose:
            print ("found", len(track.coarse_tiles_samples), "coarse tile hits")
            print ("found", len(track.pixel_samples), "pixel hits")
        
        return 

    def point_sample_tile_current(self, tile_coord, track, sample_mask):
        """
        readout.point_sample_tile_current(tile_coord,
                                          track,
                                          sample_mask)

        From the array of arrival times and charges, build a stacked array of
        charge response and time ticks.

        Parameters
        ----------
        tile_coord : tuple(int, float, float)
            The (i_tpc, x, y) (anode) coordinates of a given tile.
        track : Track object
            A track which has already had drift applied.
        sample_mask : array-like[bool]
            A boolean mask specifying which point sample charges will have
            a non-negligible induction on the tile of interest.

        Returns
        -------
        time : array-like
            Array of timestamps corresponding to the arrival of instantaneous charge.
            Shape: (N_samples,).
        charge : array-like
            Array of instantaneous charge values corresponding to time.  Shape: (N_samples,).
        label : array-like (optional)
            Array of label values corresponding to charge.  Shape: (N_samples,). Not 
            returned if truth_tracking = False.

        """
        # for now, return the entire charge with the time of arrival
        # really, should return a sparse timeseries informed by FEM
        # probably loaded from a table and interpolated
        # shape will be (2, N_point_samples, N_time_samples)
        # the first axis is (timestamps, charge mass)
        # the timestamps represent the time (relative to event t0)
        # at the beginning of a small interval (ideally, much smaller than the clock)
        # and charge is the integrated current across that interval

        # sum(result[1,:,:], dim = -1) == track.drifted_track['charge']
        
        position = track.drifted_track['position'][sample_mask]
        tpc_index = track.tpc_track['TPC_index'][sample_mask]
        charge = track.drifted_track['charge'][sample_mask]
        label = track.tpc_track['label'][sample_mask]
        time = track.drifted_track['time'][sample_mask]

        pitch = self.readout_config['coarse_tiles']['pitch']
        
        lands_on_tile_of_interest = position[:,0] - tile_coord[1] > -0.5*pitch
        lands_on_tile_of_interest *= position[:,0] - tile_coord[1] < 0.5*pitch
        lands_on_tile_of_interest *= position[:,1] - tile_coord[2] > -0.5*pitch
        lands_on_tile_of_interest *= position[:,1] - tile_coord[2] < 0.5*pitch

        # to be replaced with a model that depends on position relative to tile
        induced_charge = torch.where(lands_on_tile_of_interest,
                                     charge,
                                     torch.zeros(1),
                                     )

        if self.truth_tracking:
            return time, induced_charge, label
        else:
            return time, induced_charge
    
    def compose_tile_currents(self, time, charge, label = None):
        """
        readout.compose_tile_currents(sparse_current_series)

        Combine multiple sparse timeseries into a single dense timeseries.

        Parameters
        ----------
        time : array-like
            Array (shape: (N_samples,)) containing charge arrival time data.
            Expected array is output from readout.point_sample_tile_current.
        charge : array-like
            Array (shape: (N_samples,)) containing charge value data.
            Expected array is output from readout.point_sample_tile_current.
        label : array-like
            Array (shape: (N_samples,)) containing label data associated to charge.
            Expected array is output from readout.point_sample_tile_current.
        
        Returns
        -------
        clock_ticks : array-like
            Array of timestamps at the beginning of each clock cycle.
        induced_charge : array-like
            Array of integrated charge induced on tile within each clock cycle.
        induced_charge_by_label : array-like
            2-D array of integrated charge induced on tile within each clock cycle,
            broken down by each label category.
        unique_labels : array-like
            Array of unique_labels encountered by this tile. Indexing should be
            identical to induced_charge_by_label. 
        
        """
        try: # there is a bug where sparse_current_series is sometimes empty
            last_charge_arrival_time = torch.max(time)
            # when there is only one charge sample in a coarse cell's field
            # and it is also the earliest charge sample, n_clock_ticks is 0
            # so, add an extra clock tick to be safe
            n_clock_ticks = torch.ceil((last_charge_arrival_time - self.clock_start_time)/self.readout_config['coarse_tiles']['clock_interval']).int() + 1 
        
            arrival_time_bin_edges = torch.linspace(self.clock_start_time,
                                                    self.clock_start_time + n_clock_ticks*self.readout_config['coarse_tiles']['clock_interval'],
                                                    n_clock_ticks + 1,
                                                    )

            # find the induced charge which falls into each clock bin
            induced_charge = torchist.histogram(time,
                                                weights = charge,
                                                edges = arrival_time_bin_edges)
            if self.truth_tracking:
                # convert input labels to indices (sequential values)
                unique_labels = torch.unique(label)
                label_to_index = {int(pdg.item()): i for i, pdg in enumerate(unique_labels)}
                index_to_label = {i: pdg for pdg, i in label_to_index.items()}
                label_bin_edges = torch.arange(unique_labels.shape[0]+1)
                label_indices = torch.tensor([label_to_index[pdg.item()] for pdg in label])

                induced_charge_by_label = torchist.histogramdd(torch.stack((time,
                                                                            label_indices)).T,
                                                               weights = charge,
                                                               edges = (arrival_time_bin_edges,
                                                                        label_bin_edges))

                return arrival_time_bin_edges[:-1], induced_charge, induced_charge_by_label, unique_labels

            return arrival_time_bin_edges[:-1], induced_charge
        
        except RuntimeError:
            arrival_time_bin_edges = torch.tensor([self.clock_start_time,
                                                   self.clock_start_time + self.readout_config['coarse_tiles']['clock_interval'],
                                                   ])
            induced_charge = torch.zeros_like(arrival_time_bin_edges)
            if self.truth_tracking:
                unique_labels = torch.unique(label)
                induced_charge_by_label = torch.zeros((arrival_time_bin_edges.shape[0],
                                                       0))
            
                return arrival_time_bin_edges[:-1], induced_charge, induced_charge_by_label, unique_labels

            return arrival_time_bin_edges[:-1], induced_charge
        
    def tile_receptive_field(self, tile_coord, track, n_neighbor_tiles = 0, **kwargs):
        """
        readout.tile_receptive_field(tile_coord,
                                     track,
                                     n_neightbor_tiles = 0)
        
        For each tile, select the track samples within the tile's receptive field.

        Parameters
        ----------
        tile_coord : tuple(int, float, float)
            The (i_tpc, x, y) (anode) coordinates of a given tile.
        track : Track object
            A track which has already had drift applied.
        n_neighbor_tiles : int
            The number of adjacent tiles to include (1 means include the 8 adjacent
            and semi-adjacent tiles surrounding the provided tile coord).

        Returns
        -------
        sample_mask : array-like[bool]
            A boolean array with the same shape as the number of point samples
            contained in the `track` argument.
        
        """

        position = track.drifted_track['position']
        pitch = self.readout_config['coarse_tiles']['pitch']

        sample_mask = position[:,0] - tile_coord[1] > -(n_neighbor_tiles + 0.5)*pitch
        sample_mask *= position[:,0] - tile_coord[1] <= (n_neighbor_tiles + 0.5)*pitch
        sample_mask *= position[:,1] - tile_coord[2] > -(n_neighbor_tiles + 0.5)*pitch
        sample_mask *= position[:,1] - tile_coord[2] <= (n_neighbor_tiles + 0.5)*pitch

        return sample_mask 

    def tile_current_builder(self, track, **kwargs):
        """
        readout.tile_current_builder(track, **kwargs)

        Build the current timeseries for each tile.

        Parameters
        ----------
        track : Track obect
            Input event representation containing a populated drited_track.

        Returns
        -------
        coarse_grid_timeseries : dict{tuple: array-like}
            Dictionary of charge timeseries for tiles in detector which measure
            some charge.
        
        """
        coarse_grid_timeseries = {}
        pitch = self.readout_config['coarse_tiles']['pitch']
        for volume_name in self.detector_config['drift_volumes'].keys():
            i_tpc = self.coordinate_manager.volume_to_index[volume_name]
            tpc_mask = track.tpc_track['TPC_index'] == i_tpc
            
            min_tile = torch.tensor([-0.5*self.detector_config['drift_volumes'][volume_name]['anode_span']['width'],
                                     -0.5*self.detector_config['drift_volumes'][volume_name]['anode_span']['height'],
                                     ])
            tile_ind = torch.floor(torch.div(track.drifted_track['position'][tpc_mask,:2] - min_tile, pitch)).int()

            unique_tile_indices = torch.unique(tile_ind, dim = 0)

            for this_tile_ind in unique_tile_indices:
                tile_center = min_tile + (this_tile_ind + 0.5*torch.ones(2))*pitch
                tile_coord = (i_tpc,
                              round(float(tile_center[0]), 3),
                              round(float(tile_center[1]), 3))

                sample_mask = self.tile_receptive_field(tile_coord, track)
                tile_sample_current_series = self.point_sample_tile_current(tile_coord,
                                                                            track,
                                                                            sample_mask)
                tile_current_series = self.compose_tile_currents(*tile_sample_current_series)
                coarse_grid_timeseries[tile_coord] = tile_current_series

        return coarse_grid_timeseries

    def point_sample_pixel_current(self, pixel_coord, track, sample_mask):
        """
        readout.point_sample_pixel_current(tile_coord,
                                           track,
                                           sample_mask)

        Build a sparse timeseries for each input position.  Positions are
        relative to the pixel center.

        Parameters
        ----------
        pixel_coord : tuple(float, float)
            The (x, y) (anode) coordinates of a given pixel.
        track : Track object
            A track which has already had drift applied.
        sample_mask : array-like[bool]
            A boolean mask specifying which point sample charges will have
            a non-negligible induction on the pixel of interest.

        Returns
        -------
        time : array-like
            Array of timestamps corresponding to the arrival of instantaneous charge.
            Shape: (N_samples,).
        charge : array-like
            Array of instantaneous charge values corresponding to time.  Shape: (N_samples,).
        label : array-like
            Array of label values corresponding to charge.  Shape: (N_samples,).

        """
        # for now, return the entire charge with the time of arrival
        # really, should return a sparse timeseries informed by FEM
        # probably loaded from a table and interpolated
        # shape will be (2, N_point_samples, N_time_samples)
        # the first axis is (timestamps, charge mass)
        # the timestamps represent the time (relative to event t0)
        # at the beginning of a small interval (ideally, much smaller than the clock)
        # and charge is the integrated current across that interval

        # sum(result[1,:,:], dim = -1) == track.drifted_track['charge']
        
        position = track.drifted_track['position'][sample_mask]
        charge = track.drifted_track['charge'][sample_mask]
        label = track.tpc_track['label'][sample_mask]
        time = track.drifted_track['time'][sample_mask]
        
        pitch = self.readout_config['pixels']['pitch']

        lands_on_pixel_of_interest = position[:,0] - pixel_coord[0] > -0.5*pitch
        lands_on_pixel_of_interest *= position[:,0] - pixel_coord[0] < 0.5*pitch
        lands_on_pixel_of_interest *= position[:,1] - pixel_coord[1] > -0.5*pitch
        lands_on_pixel_of_interest *= position[:,1] - pixel_coord[1] < 0.5*pitch

        induced_charge = torch.where(lands_on_pixel_of_interest,
                                     charge,
                                     torch.zeros(1),
                                     )

        if self.truth_tracking:
            return time, induced_charge, label

        return time, induced_charge

    def compose_pixel_currents(self, coarse_cell_hit, time, charge, label = None):
        """
        readout.compose_pixel_currents(sparse_current_series, coarse_cell_hit)

        Combine multiple sparse timeseries into a single dense timeseries.

        Parameters
        ----------
        coarse_cell_hit : CoarseGridSample object
            The coarse cell hit inside which this pixel lies.
        time : array-like
            Array (shape: (N_samples,)) containing charge arrival time data.
            Expected array is output from readout.point_sample_pixel_current.
        charge : array-like
            Array (shape: (N_samples,)) containing charge value data.
            Expected array is output from readout.point_sample_pixel_current.
        label : array-like
            Array (shape: (N_samples,)) containing label data associated to charge.
            Expected array is output from readout.point_sample_pixel_current.
        
        Returns
        -------
        clock_ticks : array-like
            Array of timestamps at the beginning of each clock cycle.
        induced_charge : array-like
            Array of integrated charge induced on pixel within each clock cycle.
        induced_charge_by_label : array-like
            2-D array of integrated charge induced on pixel within each clock cycle,
            broken down by each label category.
        unique_labels : array-like
            Array of unique_labels encountered by this pixel. Indexing should be
            identical to induced_charge_by_label. 

        """
        cell_clock_start = coarse_cell_hit.coarse_measurement_time
        cell_clock_end = coarse_cell_hit.coarse_measurement_time + self.readout_config['coarse_tiles']['clock_interval']*self.readout_config['coarse_tiles']['integration_length']
        
        n_clock_ticks = int((cell_clock_end - cell_clock_start)/self.readout_config['pixels']['clock_interval'])
        arrival_time_bin_edges = torch.linspace(cell_clock_start,
                                                cell_clock_end,
                                                n_clock_ticks + 1,
                                                )

        # find the induced charge which falls into each clock bin
        induced_charge = torchist.histogram(time,
                                            weights = charge,
                                            edges = arrival_time_bin_edges)

        if self.truth_tracking:
            # convert input labels to indices (sequential values)
            unique_labels = torch.unique(label)
            label_to_index = {int(pdg.item()): i for i, pdg in enumerate(unique_labels)}
            index_to_label = {i: pdg for pdg, i in label_to_index.items()}
            label_bin_edges = torch.arange(unique_labels.shape[0]+1)
            label_indices = torch.tensor([label_to_index[pdg.item()] for pdg in label])

            induced_charge_by_label = torchist.histogramdd(torch.stack((time,
                                                                        label_indices)).T,
                                                           weights = charge,
                                                           edges = (arrival_time_bin_edges,
                                                                    label_bin_edges))
            
            return arrival_time_bin_edges[:-1], induced_charge, induced_charge_by_label, unique_labels

        return arrival_time_bin_edges[:-1], induced_charge

    def pixel_receptive_field(self, pixel_coord, track, n_neighbor_pixels = 0, **kwargs):
        """
        readout.tile_receptive_field(pixel_coord,
                                     track,
                                     n_neightbor_pixels = 0)
        
        For each tile, select the track samples within the pixel's receptive field

        Parameters
        ----------
        pixel_coord : tuple(float, float)
            The (x, y) (anode) coordinates of a given pixel.
        track : Track object
            A track which has already had drift applied.
        n_neighbor_pixels : int
            The number of adjacent pixels to include (1 means include the 8 adjacent
            and semi-adjacent pixels surrounding the provided pixel coord).

        Returns
        -------
        sample_mask : array-like[bool]
            A boolean array with the same shape as the number of point samples
            contained in the `track` argument.
        
        """

        position = track.drifted_track['position']
        pitch = self.readout_config['pixels']['pitch']

        sample_mask = position[:,0] - pixel_coord[0] > -(n_neighbor_pixels + 0.5)*pitch
        sample_mask *= position[:,0] - pixel_coord[0] <= (n_neighbor_pixels + 0.5)*pitch
        sample_mask *= position[:,1] - pixel_coord[1] > -(n_neighbor_pixels + 0.5)*pitch
        sample_mask *= position[:,1] - pixel_coord[1] <= (n_neighbor_pixels + 0.5)*pitch

        return sample_mask 
    
    def pixel_current_builder(self, track, coarse_tile_hits, **kwargs):
        """
        readout.pixel_current_builder(track, coarse_tile_hits, **kwargs)

        Build the current timeseries for each pixel.

        Parameters
        ----------
        track : Track obect
            Input event representation containing a populated drited_track.
        coarse_tile_hits : list[CoarseGridSample]
            List of found coarse hits.

        Returns
        -------
        pixel_timeseries : dict{tuple: array-like}
            Dictionary of charge timeseries for pixels in detector which measure
            some charge.
        
        """
        pixel_timeseries = {}
        
        tile_pitch = self.readout_config['coarse_tiles']['pitch']
        pixel_pitch = self.readout_config['pixels']['pitch']
        
        for this_coarse_hit in coarse_tile_hits:

            cell_tpc = this_coarse_hit.coarse_cell_tpc # may change
            cell_center_xy = this_coarse_hit.coarse_cell_pos # may change
            cell_trigger_t = this_coarse_hit.coarse_measurement_time
            
            x_bounds = [cell_center_xy[0] - 0.5*tile_pitch,
                        cell_center_xy[0] + 0.5*tile_pitch]
            y_bounds = [cell_center_xy[1] - 0.5*tile_pitch,
                        cell_center_xy[1] + 0.5*tile_pitch]
            t_bounds = [cell_trigger_t,
                        cell_trigger_t + self.readout_config['coarse_tiles']['clock_interval']*self.readout_config['coarse_tiles']['integration_length']]

            in_cell_mask = track.drifted_track['position'][:,0] >= x_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,0] < x_bounds[1]
            in_cell_mask *= track.drifted_track['position'][:,1] >= y_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,1] < y_bounds[1]
            in_cell_mask *= track.drifted_track['time'] >= t_bounds[0]
            in_cell_mask *= track.drifted_track['time'] < t_bounds[1]
            in_cell_mask *= track.tpc_track['TPC_index'] == cell_tpc

            in_cell_positions = track.drifted_track['position'][in_cell_mask]
            in_cell_charges = track.drifted_track['charge'][in_cell_mask]
            
            min_pixel = torch.tensor([x_bounds[0],
                                      y_bounds[0],
                                      ])
            n_pixels_x = int((x_bounds[1] - x_bounds[0])/pixel_pitch)
            n_pixels_y = int((y_bounds[1] - y_bounds[0])/pixel_pitch)
            pixel_volume_edges = (torch.linspace(x_bounds[0], x_bounds[1], n_pixels_x+1),
                                  torch.linspace(y_bounds[0], y_bounds[1], n_pixels_y+1))
            
            # generate a unique id for each pixel within this coarse hit
            # so that hits from 
            pixel_ind = torch.floor(torch.div(in_cell_positions[:,[0, 1]] - min_pixel, pixel_pitch)).int()
            unique_pixel_indices = torch.unique(pixel_ind, dim = 0)
            
            for this_pixel_ind in unique_pixel_indices:
                pixel_center = torch.tensor([pixel_volume_edges[i][this_pixel_ind[i]] + 0.5*pixel_pitch
                                             for i in range(2)])
                pixel_coord = (round(float(pixel_center[0]), 3),
                               round(float(pixel_center[1]), 3))

                # sample_mask = torch.all(pixel_ind == this_pixel_ind, dim = 1)
                sample_mask = self.pixel_receptive_field(pixel_coord, track)
                pixel_sample_current_series = self.point_sample_pixel_current(pixel_coord,
                                                                              track,
                                                                              sample_mask)
                pixel_current_series = self.compose_pixel_currents(this_coarse_hit,
                                                                   *pixel_sample_current_series)
                pixel_timeseries[(cell_tpc,
                                  pixel_coord[0],
                                  pixel_coord[1],
                                  cell_trigger_t)] = pixel_current_series
                                           
        return pixel_timeseries

class GAMPixModel (ReadoutModel):
    """
    GAMPixModel(readout_config)

    Implementation of hit-finding logic for a GAMPix-style pixel plane.  This
    model uses a periodic measurement scheme, with a per-pixel threshold for
    digitization and recording of accumulated charge.

    Attributes
    ----------
    readout_config : ReadoutConfig object
        Config object containing specifications for tile and pixel size, gaps,
        threshold, noise, etc.
    physics_config : PhysicsConfig object
        Config object containing physics parameters for liquid Argon.
    detector_config : DetectorConfig object
        Config object containing specifications for TPC volume position and
        orientation.
    coordinate_manager : CoordinateManager object
        Coordinate manager defined by the provided physics_config.  This is a
        helper for transforming to and from TPC-specific coordinate systems.
    clock_start_time : float
        timestamp of the earliest arriving charge bundle.  This serves as the
        first clock tick of the event readout.
    coarse_tile_hits : array-like[CoarseTileHit]
        Array of coarse tile hits populated by tile_hit_finding method
    fine_pixel_hits : array-like[CoarseTileHit]
        Array of pixel hits populated by pixel_hit_finding method

    See Also
    --------
    ReadoutModel : Parent class for readout models, in which the general scheme
        for charge simulation is implemented.
    LArPixModel : Sub-class implementing a LArPix-like hit-finding scheme.    
    
    """
    def tile_hit_finding(self, track, tile_timeseries, nonoise = False, **kwargs):
        """
        readout.tile_hit_finding(track,
                                 tile_timeseries,
                                 nonoise = False,
                                 **kwargs)

        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length).

        Parameters
        ----------
        track : Track object
            Input event representation containing a populated drifted track.
        tile_timeseries : array-like
            Array containing charge timeseries on a given tile.
        nonoise : bool
            Simulate front-end noise during the hit finding process.  Disable
            to simulate the mean behavior for this signal (useful for
            hypothesis testing).

        Returns
        -------
        hits : list[CoarseTileSample]
            List of found hits on the coarse tiles.
        
        """
        hits = []
        
        for tile_key, tile_value in tile_timeseries.items():
            if self.truth_tracking:
                time_ticks, interval_charge, interval_charge_by_label, labels = tile_value
            else:
                time_ticks, interval_charge = tile_value
                
            tile_tpc = tile_key[0]
            tile_center = (tile_key[1], tile_key[2])

            hold_length = self.readout_config['coarse_tiles']['integration_length']            
            
            # search along the bins until no more threshold crossings
            no_more_hits = False
            while not no_more_hits:
                window_charge = torch.conv_tbc(interval_charge[:,None,None],
                                               torch.ones(hold_length,1,1),
                                               bias = torch.zeros(1),
                                               pad = hold_length-1)[:,0,0]
                window_charge = window_charge[hold_length-1:]
                
                if self.truth_tracking:
                    window_charge_by_label = torch.conv_tbc(interval_charge_by_label[:,:,None],
                                                            torch.ones(hold_length,1,1),
                                                            bias = torch.zeros(1),
                                                            pad = hold_length-1)[:,:,0]
                    window_charge_by_label = window_charge_by_label[hold_length-1:]
                
                threshold = self.readout_config['coarse_tiles']['noise']*self.readout_config['coarse_tiles']['threshold_sigma']

                threshold_crossing_mask = window_charge > threshold
                threshold_crossing_mask *= interval_charge > 0

                if torch.any(threshold_crossing_mask):
                    hit_index = threshold_crossing_mask.nonzero()[0][0]
                    
                    threshold_crossing_t = time_ticks[hit_index] 

                    # is there a better way to do this?
                    threshold_crossing_z = threshold_crossing_t*self.physics_config['charge_drift']['drift_speed']
                    
                    threshold_crossing_charge = window_charge[hit_index]
                    if not nonoise:
                        threshold_crossing_charge += torch.poisson(torch.tensor(self.readout_config['coarse_tiles']['noise']).float())

                    if self.truth_tracking:
                        threshold_crossing_charge_by_label = window_charge_by_label[hit_index,:]
                        attribution_by_label = threshold_crossing_charge_by_label/threshold_crossing_charge
                    else:
                        labels = torch.zeros((0))
                        threshold_crossing_charge_by_label = torch.zeros((0))
                        attribution_by_label = torch.zeros((0))
                    
                    interval_charge[:hit_index+hold_length] = 0

                    hits.append(CoarseGridSample(tile_tpc,
                                                 tile_center,
                                                 threshold_crossing_t.item(),
                                                 threshold_crossing_z.item(),
                                                 threshold_crossing_charge.item(),
                                                 attribution_by_label.cpu().numpy(),
                                                 labels.cpu().numpy()))
                else:
                    no_more_hits = True

        track.coarse_tiles_samples = hits
        return hits 

    def pixel_hit_finding(self, track, pixel_timeseries, nonoise = False, **kwargs):
        """
        readout.pixel_hit_finding(track,
                                  tile_timeseries,
                                  nonoise = False,
                                  **kwargs)

        This method of hit finding records the charge on a pixel
        to a capacitor buffer at each time tick.
        If the total charge collected is above a threshold,
        digitize each measurement

        Parameters
        ----------
        track : Track object
            Input event representation containing a populated drifted track.
        pixel_timeseries : array-like
            Array containing charge timeseries on a given pixel.
        nonoise : bool
            Simulate front-end noise during the hit finding process.  Disable
            to simulate the mean behavior for this signal (useful for
            hypothesis testing).

        Returns
        -------
        hits : list[PixelSample]
            List of found hits on the pixel.
        
        """
        hits = []
        
        for pixel_key, pixel_value in pixel_timeseries.items():
            pixel_tpc = pixel_key[0]
            pixel_center = (pixel_key[1], pixel_key[2])
            cell_trigger_t = pixel_key[3]

            if self.truth_tracking:
                time_ticks, interval_charge, interval_charge_by_label, labels = pixel_value
            else:
                time_ticks, interval_charge = pixel_value
            
            discrim_charge = torch.sum(interval_charge)+torch.poisson(torch.tensor(self.readout_config['pixels']['noise']).float())
            threshold = self.readout_config['pixels']['noise']*self.readout_config['pixels']['threshold_sigma']
            if discrim_charge > threshold:

                if self.truth_tracking:
                    attribution_by_label = interval_charge_by_label.T/interval_charge
                    attribution_by_label = torch.where(torch.isfinite(attribution_by_label),
                                                       attribution_by_label,
                                                       torch.zeros_like(attribution_by_label)).T
                else:
                    labels = torch.zeros((0,))
                    attribution_by_label = torch.zeros((interval_charge.shape[0], 0))
                    
                if not nonoise:
                    interval_charge += torch.poisson(self.readout_config['pixels']['noise']*torch.ones_like(interval_charge))
                
                for this_timestamp, this_interval_charge, this_attribution_by_label in zip(time_ticks, interval_charge, attribution_by_label):
                    this_z = this_timestamp*self.physics_config['charge_drift']['drift_speed'] 

                    hits.append(PixelSample(pixel_tpc,
                                            pixel_center,
                                            this_timestamp.item(),
                                            this_z.item(),
                                            this_interval_charge.item(),
                                            this_attribution_by_label.cpu().numpy(),
                                            labels.cpu().numpy(),
                                            ))
        track.pixel_samples = hits
        return hits 

class LArPixModel (ReadoutModel):
    """
    LArPixModel(readout_config)

    Implementation of hit-finding logic for a LArPix-style pixel plane.  This
    model uses a threshold and buffer measurement scheme, where charge is allowed
    to accumulate for a fixed number of clock cycles after threshold crossing
    before digitization and flushing.

    Attributes
    ----------
    readout_config : ReadoutConfig object
        Config object containing specifications for tile and pixel size, gaps,
        threshold, noise, etc.
    clock_start_time : float
        timestamp of the earliest arriving charge bundle.  This serves as the
        first clock tick of the event readout.
    physics_config : PhysicsConfig object
        Config object containing physics parameters for liquid Argon.
    detector_config : DetectorConfig object
        Config object containing specifications for TPC volume position and
        orientation.
    coordinate_manager : CoordinateManager object
        Coordinate manager defined by the provided physics_config.  This is a
        helper for transforming to and from TPC-specific coordinate systems.
    coarse_tile_hits : array-like[CoarseTileHit]
        Array of coarse tile hits populated by tile_hit_finding method
    fine_pixel_hits : array-like[CoarseTileHit]
        Array of pixel hits populated by pixel_hit_finding method

    See Also
    --------
    ReadoutModel : Parent class for readout models, in which the general scheme
        for charge simulation is implemented.
    GAMPixModel : Sub-class implementing a GAMPix-like hit-finding scheme.
    
    """
    def tile_hit_finding(self, track, tile_timeseries, nonoise = False, **kwargs):
        """
        readout.tile_hit_finding(track,
                                 tile_timeseries,
                                 nonoise = False,
                                 **kwargs)

        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        
        Parameters
        ----------
        track : Track object
            Input event representation containing a populated drifted track.
        tile_timeseries : array-like
            Array containing charge timeseries on a given tile.
        nonoise : bool
            Simulate front-end noise during the hit finding process.  Disable
            to simulate the mean behavior for this signal (useful for
            hypothesis testing).

        Returns
        -------
        hits : list[CoarseTileSample]
            List of found hits on the coarse tiles.
        
        """
        # TODO: fix logic for integration_length > 1
        hits = []
        
        for tile_center, timeseries in tile_timeseries.items():
            time_ticks, interval_charge = timeseries
            
            hold_length = self.readout_config['coarse_tiles']['integration_length']            
            
            # search along the bins until no more threshold crossings
            no_more_hits = False
            while not no_more_hits:
                window_charge = torch.conv_tbc(interval_charge[:,None,None],
                                               torch.ones(hold_length,1,1),
                                               bias = torch.zeros(1),
                                               pad = hold_length-1)[:,0,0]
                window_charge = window_charge[hold_length-1:]
                
                threshold = self.readout_config['coarse_tiles']['noise']*self.readout_config['coarse_tiles']['threshold_sigma']

                threshold_crossing_mask = window_charge > threshold
                threshold_crossing_mask *= interval_charge > 0

                if torch.any(threshold_crossing_mask):
                    hit_index = threshold_crossing_mask.nonzero()[0][0]
                    
                    threshold_crossing_t = time_ticks[hit_index]

                    # is there a better way to do this?
                    threshold_crossing_z = threshold_crossing_t*self.physics_config['charge_drift']['drift_speed'] 
                
                    threshold_crossing_charge = window_charge[hit_index]
                    if not nonoise:
                        threshold_crossing_charge += torch.poisson(torch.tensor(self.readout_config['coarse_tiles']['noise']).float())

                    interval_charge[:hit_index+hold_length] = 0

                    hits.append(CoarseGridSample(tile_center,
                                                 threshold_crossing_t.item(),
                                                 threshold_crossing_z.item(),
                                                 threshold_crossing_charge.item()))
                else:
                    no_more_hits = True

        track.coarse_tiles_samples = hits
        return hits 

    def pixel_hit_finding(self, track, pixel_timeseries, nonoise = False, **kwargs):
        """
        readout.pixel_hit_finding(track,
                                  tile_timeseries,
                                  nonoise = False,
                                  **kwargs)

        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        
        Parameters
        ----------
        track : Track object
            Input event representation containing a populated drifted track.
        pixel_timeseries : array-like
            Array containing charge timeseries on a given pixel.
        nonoise : bool
            Simulate front-end noise during the hit finding process.  Disable
            to simulate the mean behavior for this signal (useful for
            hypothesis testing).

        Returns
        -------
        hits : list[PixelSample]
            List of found hits on the pixel.
        
        """
        hits = []
        
        for pixel_key, timeseries in pixel_timeseries.items():
            pixel_center = (pixel_key[0], pixel_key[1])
            cell_trigger_t = pixel_key[2]

            time_ticks, interval_charge = timeseries

            hold_length = self.readout_config['pixels']['integration_length']            

            # search along the bins until no more threshold crossings
            no_hits = False
            while not no_hits:
                window_charge = torch.conv_tbc(interval_charge[:,None,None],
                                               torch.ones(hold_length,1,1),
                                               bias = torch.zeros(1),
                                               pad = hold_length-1)[:,0,0]
                window_charge = window_charge[hold_length-1:]

                threshold = self.readout_config['pixels']['noise']*self.readout_config['pixels']['threshold_sigma']
                threshold_crossing_mask = window_charge > threshold
                threshold_crossing_mask *= interval_charge > 0
                
                if torch.any(threshold_crossing_mask):
                    hit_index = threshold_crossing_mask.nonzero()[0][0]

                    threshold_crossing_t = time_ticks[hit_index]

                    # is there a better way to do this?
                    threshold_crossing_z = threshold_crossing_t*self.physics_config['charge_drift']['drift_speed'] 

                    threshold_crossing_charge = window_charge[hit_index]

                    if not nonoise:
                        # add quiescent noise
                        threshold_crossing_charge += torch.poisson(torch.tensor(self.readout_config['pixels']['noise']).float())

                    interval_charge[:hit_index+hold_length] = 0

                    hits.append(PixelSample(pixel_center,
                                            threshold_crossing_t.item(),
                                            threshold_crossing_z.item(),
                                            threshold_crossing_charge.item()))
                else:
                    no_hits = True

        track.pixel_samples = hits
        return hits 

class DetectorModel:
    """
    DetectorModel(detector_params,
                  physics_params,
                  readout_params)

    Detector model class.  This class defines the overall flow of simulation for
    charges within the detector volume.

    Attributes
    ----------
    detector_params : DetectorConfig
    physics_params : PhysicsConfig
    readout_params : ReadoutConfig
    readout_model : ReadoutModel object
    
    """
    def __init__(self,
                 detector_params = default_detector_params,
                 physics_params = default_physics_params,
                 readout_params = default_readout_params,
                 truth_tracking = True):

        self.detector_params = detector_params
        self.physics_params = physics_params
        self.readout_params = readout_params

        self.coordinate_manager = CoordinateManager(detector_params)
        
        self.readout_model = GAMPixModel(readout_config = readout_params,
                                         physics_config = physics_params,
                                         detector_config = detector_params,
                                         truth_tracking = truth_tracking,
                                         )
        # self.readout_model = LArPixModel(readout_params)

        self.truth_tracking = truth_tracking
 
    def simulate(self, track, **kwargs):
        """
        detector.simulate(track, **kwargs)

        Perform drift and readout simulations for the input Track object.

        Parameters
        ----------
        track : Track obect
            Event data provided by an input parser or a generator.
        
        """
        self.drift(track, **kwargs)
        self.readout(track, **kwargs)
        
    def drift(self, sampled_track, **kwargs):
        """
        detector.drift(sampled_track, **kwargs)

        Drift the charge samples from their input position
        to the anode position as defined by detector_params.
        Save the drifted positions to the track object.

        Parameters
        ----------
        track : Track obect
            Event data provided by an input parser or a generator.
        
        """
        # first, translate the raw track position into tpc coordinates
        # this translates from (x, y, z) in experimental system to
        # (i_TPC, x, y, z) in the coordinate system of the appropriate TPC
        # in the case when there are overlapping drift volumes (this should
        # be avoided), those charges will have an entry for each volume
        self.coordinate_manager.generate_tpc_coords(sampled_track)
        
        # z-component is the distance to the anode
        drift_time = sampled_track.tpc_track['position'][:,2]/self.physics_params['charge_drift']['drift_speed'] # s

        # D according to https://lar.bnl.gov/properties/trans.html#diffusion-l
        # sigma = sqrt(2*D*t)

        # use the nominal drift time to calculate diffusion
        # then, add the appropriate arrival time dispersion later
        sigma_transverse = torch.sqrt(2*self.physics_params['charge_drift']['diffusion_transverse']*drift_time)
        sigma_longitudinal = torch.sqrt(2*self.physics_params['charge_drift']['diffusion_longitudinal']*drift_time)

        diffusion_sigma = torch.stack((sigma_transverse,
                                       sigma_transverse,
                                       sigma_longitudinal,
                                       )).T

        drifted_positions = torch.normal(sampled_track.tpc_track['position'],
                                         diffusion_sigma)

        # charge is diminished by attenuation
        drifted_charges = sampled_track.tpc_track['charge']*torch.exp(-drift_time/self.physics_params['charge_drift']['electron_lifetime'])

        # add dispersion to the arrival of charge due to longitudinal diffusion
        time_dispersion = (drifted_positions[:, 2] - sampled_track.tpc_track['position'][:, 2])/self.physics_params['charge_drift']['drift_speed'] 
        
        arrival_time = drift_time + sampled_track.tpc_track['time'] + time_dispersion
            
        # might also include a sub-sampling step?
        # in case initial sampling is not fine enough

        sampled_track.drifted_track = {'position': drifted_positions,
                                       'charge': drifted_charges,
                                       'time': arrival_time}
        return 

    def readout(self, drifted_track, **kwargs):
        """
        detector.drift(drifted_track, **kwargs)

        Apply the readout simulation to the drifted track.  This is defined
        by the readout_model attribute.
        
        Parameters
        ----------
        drifted_track : Track obect
            Event data provided by an input parser or a generator.  This object
            should have already had drift simulation applied so that drifted_track
            attribute is defined and non-empty.
        
        """
        self.readout_model.electronics_simulation(drifted_track, **kwargs)

        return
