from gampixpy import config

import numpy as np

NULL_EVENT = -1
NULL_LABEL = -9999

def dtype_factory(readout_config = config.default_readout_params):
    """
    dtype_factory(readout_config)

    Create the appropriate record dtypes for a given readout config.

    Parameters
    ----------
    readout_config : ReadoutConfig
        A dict-like object containing input and derived parameters for
        readout electronics simulation.

    Returns
    -------
    tile_dtype, pixel_dtype : np.dtype, np.dtype
        dtype objects specifying how record info is stored in a binary array.
    """
    
    truth_tracking = readout_config['truth_tracking']['enabled']
    n_labels = readout_config['truth_tracking']['n_labels']
    tile_waveform_length = readout_config['coarse_tiles']['integration_length']
    
    if truth_tracking:
        tile_dtype = np.dtype([("event id", "u4"),
                               ("tile trigger id", "u4"),
                               ("tile tpc", "u4"),
                               ("tile x", "f4"),
                               ("tile y", "f4"),
                               ("trig z", "f4"),
                               ("trig t", "f4"),
                               ("waveform", "f4",
                                tile_waveform_length),
                               ("attribution", "f4",
                                (tile_waveform_length,
                                 n_labels)),
                               ("label", "i4", n_labels),
                               ],
                              align = True)

        pixel_dtype = np.dtype([("event id", "u4"),
                                ("tile trigger id", "u4"),
                                ("pixel tpc", "u4"),
                                ("pixel x", "f4"),
                                ("pixel y", "f4"),
                                ("trig z", "f4"),
                                ("trig t", "f4"),
                                ("waveform", "f4",
                                 tile_waveform_length),
                                ("attribution", "f4",
                                 (tile_waveform_length,
                                  n_labels)),
                                ("label", "i4", n_labels),
                                ],
                               align = True)
    else:
        tile_dtype = np.dtype([("event id", "u4"),
                               ("tile trigger id", "u4"),
                               ("tile tpc", "u4"),
                               ("tile x", "f4"),
                               ("tile y", "f4"),
                               ("trig z", "f4"),
                               ("trig t", "f4"),
                               ("waveform", "f4",
                                tile_waveform_length),
                               ],
                              align = True)

        pixel_dtype = np.dtype([("event id", "u4"),
                                ("tile trigger id", "u4"),
                                ("pixel tpc", "u4"),
                                ("pixel x", "f4"),
                                ("pixel y", "f4"),
                                ("trig z", "f4"),
                                ("trig t", "f4"),
                                ("waveform", "f4",
                                 tile_waveform_length),
                                ],
                               align = True)
        
    return tile_dtype, pixel_dtype

def pixel_record_factory(config_manager = config.default_config_manager):
    """
    pixel_record_factory(config_manager)

    Create the appropriate record class for a given readout config.

    Parameters
    ----------
    config_manager : ConfigManager
        A container object for managing multiple configuration dicts.

    Returns
    -------
    PixelRecord : class
        class specifying the PixelRecord data container.

    """

    readout_config = config_manager.readout_config
    physics_config = config_manager.physics_config

    truth_tracking = readout_config['truth_tracking']['enabled']
    n_labels = readout_config['truth_tracking']['n_labels']
    tile_waveform_length = readout_config['coarse_tiles']['integration_length']

    class PixelRecord:
        """
        PixelRecord(pixel_tpc,
                    pixel_pos,
                    tile_trigger_id,
                    trigger_timestamp,
                    trigger_depth,
                    timeticks,
                    waveform,
                    attribution,
                    labels)

        Data container class for pixel samples.

        Attributes
        ----------
        pixel_tpc : int
            TPC index of pixel.
        pixel_pos : tuple(float, float)
            Position in anode coordinates (x, y) of pixel center.
        tile_trigger_id : int
            ID corresponding to the coarse tile trigger.  Unique per event.
        trigger_time : float
            Timestamp associated with beginning of measurement.
            Depending on the hit finding method used, this may
            be the time of theshold crossing or the time of digitization.
        trigger_depth : float
            Estimated depth assiciated with this hit.  This is usually just
            arrival_time*v_drift, and so ignores some details of hit finding.
        timeticks : array(float)
            Clock values corresponding to each discrete measurement in the waveform.
        waveform : array(float)
            Measured charge series (or correlate) for this trigger.
        attribution : array(float, float)
            Fractional attribution for each measurement in the waveform,
            divided into the specific label classes.
        labels : array(int)
            Label classes corresponding to each column of attribution array.
        """

        _truth_tracking = truth_tracking
        _n_labels = n_labels
        _tile_waveform_length = tile_waveform_length
        
        def __init__(self,
                     pixel_tpc,
                     pixel_pos,
                     tile_trigger_id,
                     trigger_time,
                     trigger_depth,
                     timeticks,
                     waveform,
                     attribution,
                     labels):
            self.pixel_tpc = pixel_tpc
            self.pixel_pos = pixel_pos
            self.tile_trigger_id = tile_trigger_id
            self.trigger_time = trigger_time
            self.trigger_depth = trigger_depth
            self.timeticks = timeticks
            self.waveform = waveform

            if self._truth_tracking:
                # save the _n_label highest contributing labels
                # if tere are fewer than _n_labels, label is 0
                # and fraction is 0
                self.attribution = np.zeros((self._tile_waveform_length,
                                             self._n_labels))
                self.labels = NULL_LABEL*np.ones(self._n_labels)
                total_charge_by_label = np.sum(attribution*waveform[:,None], axis = 0)
                for i, sorted_ind in enumerate(np.argsort(total_charge_by_label)[::-1]):
                    if i < self._n_labels:
                        self.attribution[:,i] = attribution[:,sorted_ind]
                        self.labels[i] = labels[sorted_ind]
            else:
                self.attribution = attribution
                self.labels = labels
                        
        @classmethod
        def from_numpy(cls, array):
            trigger_time = array['trig t'],
            dt = readout_config['pixels']['clock_interval']
            timeticks = trigger_time + dt*np.arange(tile_waveform_length)

            trigger_depth = array['trig z']
            v_drift = physics_config['charge_drift']['drift_speed']
            depthticks = trigger_depth + dt*np.arange(tile_waveform_length)*v_drift

            return cls(array['pixel tpc'],
                       [array['pixel x'], array['pixel y']],
                       array['tile trigger id'],
                       trigger_time,
                       depthticks,
                       timeticks,
                       array['waveform'],
                       array['attribution'],
                       array['label'],
                       )

    return PixelRecord

def tile_record_factory(config_manager = config.default_config_manager):
    """
    tile_record_factory(config_manager)

    Create the appropriate record class for a given readout config.

    Parameters
    ----------
    config_manager : ConfigManager
        A container object for managing multiple configuration dicts.

    Returns
    -------
    TileRecord : class
        class specifying the TileRecord data container.

    """

    readout_config = config_manager.readout_config
    physics_config = config_manager.physics_config

    truth_tracking = readout_config['truth_tracking']['enabled']
    if truth_tracking:
        n_labels = readout_config['truth_tracking']['n_labels']
    else:
        n_labels = 0

    tile_waveform_length = readout_config['coarse_tiles']['integration_length']

    class TileRecord:
        """
        TileRecorde(coarse_cell_tpc,
                    coarse_cell_pos,
                    tile_trigger_id,
                    trigger_timestamp,
                    trigger_depth,
                    timeticks,
                    waveform,
                    attribution,
                    labels)

        Data container class for coarse tile samples.

        Attributes
        ----------
        tile_tpc : int
            TPC index of coarse cell.
        tile_pos : tuple(float, float)
            Position in anode coordinates (x, y) of the tile center.
        tile_trigger_id : int
            ID corresponding to the coarse tile trigger.  Unique per event.
        trigger_time : float
            Timestamp associated with beginning of measurement.
            Depending on the hit finding method used, this may be
            the time of theshold crossing or the time of digitization.
        trigger_depth : float
            Estimated depth assiciated with this hit.  This is usually just
            arrival_time*v_drift, and so ignores some details of hit finding.
        timeticks : array(float)
            Clock values corresponding to each discrete measurement in the waveform.
        waveform : array(float)
            Measured charge series (or correlate) for this trigger.
        attribution : array(float, float)
            Fractional attribution for each measurement in the waveform,
            divided into the specific label classes.
        labels : array(int)
            Label classes corresponding to each column of attribution array.
        """

        _truth_tracking = truth_tracking
        _n_labels = n_labels
        _tile_waveform_length = tile_waveform_length

        def __init__(self,
                     tile_tpc,
                     tile_pos,
                     tile_trigger_id,
                     trigger_time,
                     trigger_depth,
                     timeticks,
                     waveform,
                     attribution,
                     labels):
            self.tile_tpc = tile_tpc
            self.tile_pos = tile_pos
            self.tile_trigger_id = tile_trigger_id
            self.trigger_time = trigger_time
            self.trigger_depth = trigger_depth
            self.timeticks = timeticks
            self.waveform = waveform

            # save the _n_label highest contributing labels
            # if tere are fewer than _n_labels, label is 0
            # and fraction is 0
            
            if self._truth_tracking:
                self.attribution = np.zeros((self._tile_waveform_length,
                                             self._n_labels))
                self.labels = NULL_LABEL*np.ones(self._n_labels)
                total_charge_by_label = np.sum(attribution*waveform[:,None], axis = 0)
                for i, sorted_ind in enumerate(np.argsort(total_charge_by_label)[::-1]):
                    if i < self._n_labels:
                        self.attribution[:,i] = attribution[:,sorted_ind]
                        self.labels[i] = labels[sorted_ind]
            else:
                self.attribution = attribution
                self.labels = labels
                        
        @classmethod
        def from_numpy(cls, array):
            trigger_time = array['trig t'],
            dt = readout_config['coarse_tiles']['clock_interval']
            timeticks = trigger_time + dt*np.arange(waveform.shape[0])
            return cls(array['tile tpc'],
                       [array['tile x'], array['tile y']],
                       array['tile trigger id'],
                       trigger_time,
                       array['trig z'],
                       timeticks,
                       array['waveform'],
                       array['attribution'],
                       array['label'],
                       )

    return TileRecord
