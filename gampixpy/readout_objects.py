import numpy as np

N_LABELS_COARSE = 3
N_LABELS_PIX = 3

coarse_tile_dtype = np.dtype([("event id", "u4"),
                              ("tile tpc", "u4"),
                              ("tile x", "f4"),
                              ("tile y", "f4"),
                              ("hit z", "f4"),
                              ("hit t", "f4"),
                              ("hit charge", "f4"),
                              ("attribution", "f4", N_LABELS_COARSE),
                              ("label", "i4", N_LABELS_COARSE),
                              ],
                             align = True)
pixel_dtype = np.dtype([("event id", "u4"),
                        ("pixel tpc", "u4"),
                        ("pixel x", "f4"),
                        ("pixel y", "f4"),
                        ("hit z", "f4"),
                        ("hit t", "f4"),
                        ("hit charge", "f4"),
                        ("attribution", "f4", N_LABELS_PIX),
                        ("label", "i4", N_LABELS_PIX),
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
                 hit_measurement,
                 attribution,
                 labels):
        self.pixel_tpc = pixel_tpc
        self.pixel_pos = pixel_pos
        self.hit_timestamp = hit_timestamp
        self.hit_depth = hit_depth
        self.hit_measurement = hit_measurement

        # save the N_LABELS_PIX highest contributing labels
        # if tere are fewer than N_LABELS_PIX, label is -1
        # and fraction is 0
        self.attribution = np.zeros(N_LABELS_PIX)
        self.labels = -1*np.ones(N_LABELS_PIX)
        for i, sorted_ind in enumerate(np.argsort(attribution)[::-1]):
            self.attribution[i] = attribution[sorted_ind]
            self.labels[i] = labels[sorted_ind]

class CoarseGridSample:
    """
    CoarseGridSample(coarse_cell_tpc,
                     coarse_cell_pos,
                     hit_timestamp,
                     hit_depth,
                     hit_measurement,
                     attribution,
                     labels)

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
                 coarse_cell_measurement,
                 attribution,
                 labels):
        self.coarse_cell_tpc = coarse_cell_tpc
        self.coarse_cell_pos = coarse_cell_pos
        self.coarse_measurement_time = measurement_time
        self.coarse_measurement_depth = measurement_depth
        self.coarse_cell_measurement = coarse_cell_measurement

        # save the N_LABELS_COARSE highest contributing labels
        # if tere are fewer than N_LABELS_COARSE, label is 0
        # and fraction is 0
        self.attribution = np.zeros(N_LABELS_COARSE)
        self.labels = np.zeros(N_LABELS_COARSE)
        for i, sorted_ind in enumerate(np.argsort(attribution)[::-1]):
            self.attribution[i] = attribution[sorted_ind]
            self.labels[i] = labels[sorted_ind]
