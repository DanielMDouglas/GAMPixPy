import numpy as np

coarse_tile_dtype = np.dtype([("event id", "u4"),
                              ("tile tpc", "u4"),
                              ("tile x", "f4"),
                              ("tile y", "f4"),
                              ("hit z", "f4"),
                              ("hit t", "f4"),
                              ("hit charge", "f4"),
                              ],
                             align = True)
pixel_dtype = np.dtype([("event id", "u4"),
                        ("pixel tpc", "u4"),
                        ("pixel x", "f4"),
                        ("pixel y", "f4"),
                        ("hit z", "f4"),
                        ("hit t", "f4"),
                        ("hit charge", "f4"),
                        ],
                       align = True)

class PixelSample:
    """
    PixelSample(pixel_tpc,
                pixel_pos,
                hit_timestamp,
                hit_depth,
                hit_measurement)

    Data container class for pixel samples.

    Attributes
    ----------
    pixel_tpc : int
        TPC index of pixel.
    pixel_pos : tuple(float, float)
        Position in anode coordinates (x, y) of pixel center.
    hit_timestamp : float
        Timestamp associated with hit.  Depending on the hit finding
        method used, this may be the time of theshold crossing or the
        time of digitization.
    hit_depth : float
        Estimated depth assiciated with this hit.  This is usually just
        arrival_time*v_drift, and so ignores some details of hit finding.
    hit_measurement : float
        Measured charge (or correlate) for this hit.
    """
    def __init__(self,
                 pixel_tpc,
                 pixel_pos,
                 hit_timestamp,
                 hit_depth,
                 hit_measurement):
        self.pixel_tpc = pixel_tpc
        self.pixel_pos = pixel_pos
        self.hit_timestamp = hit_timestamp
        self.hit_depth = hit_depth
        self.hit_measurement = hit_measurement

class CoarseGridSample:
    """
    CoarseGridSample(coarse_cell_tpc,
                     coarse_cell_pos,
                     hit_timestamp,
                     hit_depth,
                     hit_measurement)

    Data container class for coarse tile samples.

    Attributes
    ----------
    coarse_cell_tpc : int
        TPC index of coarse cell.
    coarse_cell_pos : tuple(float, float)
        Position in anode coordinates (x, y) of the tile center.
    coarse_measurement_time : float
        Timestamp associated with hit.  Depending on the hit finding
        method used, this may be the time of theshold crossing or the
        time of digitization.
    measurement_depth : float
        Estimated depth assiciated with this hit.  This is usually just
        arrival_time*v_drift, and so ignores some details of hit finding.
    coarse_cell_measurement : float
        Measured charge (or correlate) for this hit.
    """
    def __init__(self,
                 coarse_cell_tpc,
                 coarse_cell_pos,
                 measurement_time,
                 measurement_depth,
                 coarse_cell_measurement):
        self.coarse_cell_tpc = coarse_cell_tpc
        self.coarse_cell_pos = coarse_cell_pos
        self.coarse_measurement_time = measurement_time
        self.coarse_measurement_depth = measurement_depth
        self.coarse_cell_measurement = coarse_cell_measurement
