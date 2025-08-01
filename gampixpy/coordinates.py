import torch

class CoordinateManager:
    """
    CoordinateManager()

    Class for transforming external coordinate systems into the internal
    coordinate system.

    Attributes
    ----------
    
    """
    def __init__(self, detector_config):
        self.detector_config = detector_config

        # dictionaries for translating from volume name stings to an internal index
        keylist = list(self.detector_config['drift_volumes'].keys())
        keylist.sort()
        self.index_to_volume = {i: volume_name
                                for i, volume_name
                                in enumerate(keylist)}
        self.volume_to_index = {volume_name: i
                                for i, volume_name
                                in self.index_to_volume.items()}
        return
        
    def to_experiment_coords(self, coords, tpc_index):
        """
        coord_manager.to_external_coords(vector)

        Transform the input vector from the internal coordinate system
        to the external (experimenatal) coordinate system specified in
        the detector config to the internal system.

        Parameters
        ----------
        coords : array-like
            An (N,3) array of (x, y, z) coordinates in the per-TPC system
        tpc_index : array-like
            A (N,) array of TPC indices, indicating which origin is used
            for each point sample.

        Returns
        -------
        experiment_coords : array-like
            an (N,3) array of (x, y, z) coordinates in the external
            reference frame.

        See Also
        --------
        to_internal_coords : inverse function for transforming from external
            coordinates to internal coordinates.
        """

        exp_coords = torch.empty(coords.shape)
        for volume_name, volume_dict in self.detector_config['drift_volumes'].items():
            tpc_origin = volume_dict['anode_center']

            subset_mask = tpc_index == self.volume_to_index[volume_name]
            coord_subset = coords[subset_mask]

            exp_coord_subset = tpc_origin + \
                coord_subset[:,0,None]*volume_dict['anode_horizontal'][None,:] + \
                coord_subset[:,1,None]*volume_dict['anode_vertical'][None,:] + \
                coord_subset[:,2,None]*volume_dict['drift_axis'][None,:]

            exp_coords[subset_mask] = exp_coord_subset

        return exp_coords

    def to_tpc_coords(self, exp_coords):
        """
        coord_manager.to_internal_coords(vector)

        Transform the input vector from the external coordinate system
        specified in the detector config to the internal system.

        Parameters
        ----------
        exp_coords : array
            an (N, 3) array of (x, y, z) coordinates in the experimental
            reference frame.
        
        Returns
        -------
        tpc_coords : array
            an (N, 4) array of (i_TPC, x, y, z) coordinates in the TPC
            reference frame.

        See Also
        --------
        to_experiment_coords : inverse function for transforming from external
            coordinates to internal coordinates.
        """

        tpc_coords = torch.empty((0,4))
        for volume_name, volume_dict in self.detector_config['drift_volumes'].items():
            # choose an arbitrary corner.  Which one doesn't matter, but it should be
            # between 0 and 7 (a rectangular prism has 8 vertices)
            reference_corner_index = 0
            corners = volume_dict['corners']
            reference_corner = corners[reference_corner_index]
            connected_corners = corners[volume_dict['connectivity'][reference_corner_index]]

            leg_vec = connected_corners - reference_corner
            leg_dists = torch.linalg.norm(leg_vec, axis = 1)
            # project each point onto each leg, then normalize by the leg length
            extent = torch.inner(exp_coords - reference_corner, leg_vec)/leg_dists**2
            # the points which are within the drift volume are ones where the extent
            # along each leg is between 0 and 1
            keep_mask = torch.all(extent >= 0, axis = 1)*torch.all(extent <= 1, axis = 1)

            # project into TPC coordinates
            # the origin in each TPC is the anode_center, given in the detector config
            anode_disp = exp_coords[keep_mask] - volume_dict['anode_center']
            tpc_coords_subset = torch.stack([self.volume_to_index[volume_name]*torch.ones(anode_disp.shape[0]),
                                             torch.inner(anode_disp, volume_dict['anode_horizontal']),
                                             torch.inner(anode_disp, volume_dict['anode_vertical']),
                                             torch.inner(anode_disp, volume_dict['drift_axis'])]).T

            tpc_coords = torch.concatenate([tpc_coords,
                                            tpc_coords_subset])

        return tpc_coords

    def generate_tpc_coords(self, track):
        """
        coord_manager.generate_tpc_coords(vector)

        Transform the raw charge cloud from an input track from
        the experimental coordinate system to per-TPC coordinates.
        This function populates the 'tpc_track' attribute of the
        Track object.

        Parameters
        ----------
        track : Track object
            Event data provided by an input parser or a generator.
        
        Notes
        -----
        Populates the input track with a 'tpc_track' dict with the transformed
            coordinates and properly masked charge and timestamps.
        
        See Also
        --------
        to_experiment_coords : inverse function for transforming from external
            coordinates to internal coordinates.
        """
        
        tpc_index = torch.empty((0,), dtype = torch.uint8)
        tpc_coords = torch.empty((0,3))
        tpc_time = torch.empty((0,))
        tpc_charge = torch.empty((0,))
        for volume_name, volume_dict in self.detector_config['drift_volumes'].items():
            # choose an arbitrary corner.  Which one doesn't matter, but it should be
            # between 0 and 7 (a rectangular prism has 8 vertices)
            reference_corner_index = 0
            reference_corner = volume_dict['corners'][reference_corner_index]
            connected_corners = volume_dict['corners'][volume_dict['connectivity'][reference_corner_index]]

            leg_vec = connected_corners - reference_corner
            leg_dists = torch.linalg.norm(leg_vec, axis = 1)
            # project each point onto each leg, then normalize by the leg length
            extent = torch.inner(track.raw_track['position'] - reference_corner, leg_vec)/leg_dists**2
            # the points which are within the drift volume are ones where the extent
            # along each leg is between 0 and 1
            keep_mask = torch.all(extent >= 0, axis = 1)*torch.all(extent <= 1, axis = 1)

            # project into TPC coordinates
            # the origin in each TPC is the anode_center, given in the detector config
            anode_disp = track.raw_track['position'][keep_mask] - volume_dict['anode_center']
            tpc_coords_subset = torch.stack([torch.inner(anode_disp, volume_dict['anode_horizontal']),
                                             torch.inner(anode_disp, volume_dict['anode_vertical']),
                                             torch.inner(anode_disp, volume_dict['drift_axis'])]).T

            tpc_index_subset = self.volume_to_index[volume_name]*torch.ones(anode_disp.shape[0])
            tpc_time_subset = track.raw_track['time'][keep_mask]
            tpc_charge_subset = track.raw_track['charge'][keep_mask]
            
            tpc_index = torch.concatenate([tpc_index,
                                           tpc_index_subset])
            tpc_coords = torch.concatenate([tpc_coords,
                                            tpc_coords_subset])
            tpc_time = torch.concatenate([tpc_time,
                                          tpc_time_subset])
            tpc_charge = torch.concatenate([tpc_charge,
                                            tpc_charge_subset])

        track.tpc_track = {'TPC_index': tpc_index,
                           'position': tpc_coords,
                           'time': tpc_time,
                           'charge': tpc_charge,
                           }

    def to_tpc_indices(self, coords):
        """
        coord_manager.to_internal_coords(vector)

        Transform the input vector from the external coordinate system
        specified in the detector config to the internal system.

        Parameters
        ----------

        Returns
        -------

        See Also
        --------
        to_internal_coords : inverse function for transforming from external
            coordinates to internal coordinates.
        """
        # is this a useful function?  Implement at some point...
        return coords
