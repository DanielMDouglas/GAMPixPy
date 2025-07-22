from gampixpy

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

        self.rotation = 0
        
        return
        
    def to_experiment_coords(self, coords):
        """
        coord_manager.to_external_coords(vector)

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
        return coords

    def to_internal_coords(self, coords):
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
        return coords
