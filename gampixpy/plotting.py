import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import SLACplots

class EventDisplay:
    """
    EventDisplay(track)

    An event display object is a general-purpose manager for building
    3D plots of a Track object in various states of simulation.

    Parameters
    ----------
    track : Track object
        The general data container object used by gampixpy.  This may
        be at any stage of simulation.

    Attributes
    ----------
    track_object : Track
        Internal pointer to the input track object.
    fig : matplotlib.figure.Figure object
        Figure object.
    ax : matplotlib.axes._axes.Axes object
        Current (3D) axes of the figure.
    coarse_tile_hit_kwargs : dict
        Keyword args for plotting coarse hit boxes.
    pixel_hit_kwargs : dict
        Keyword args for plotting pixel hit boxes.

    Examples
    --------
    >>> evd = EventDisplay(event_data)
    >>> evd.plot_drifted_track_timeline(alpha = 0) # can also pass kwargs to plt.scatter
    >>> evd.plot_coarse_tile_measurement_timeline(readout_config) # plot tile hits
    >>> evd.plot_pixel_measurement_timeline(readout_config) # plot pixel hits
    >>> evd.show()

    """
    MAX_POINTS_PLOTTED = 1e3
    def __init__(self, track):
        self.track_object = track
        self._init_fig()

        # self.coarse_tile_hit_kwargs = dict(facecolors = 'cyan',
        self.coarse_tile_hit_kwargs = dict(facecolors=SLACplots.stanford.palo_verde,
                                           linewidths=1,
                                           edgecolors=SLACplots.SLACblue,
                                           alpha = 0.25)
        self.pixel_hit_kwargs = dict(facecolors=SLACplots.stanford.illuminating,
                                     linewidths=1,
                                     edgecolors=SLACplots.stanford.illuminating,
                                     alpha = 0.5)
        
    def _init_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')

    def remove_guidelines(self):
        """
        evd.remove_guidelines()

        Removes the axis grid lines, ticks, and labels.

        Parameters
        ----------
        None

        """
        self.ax.axis('off')
        
    def equal_aspect(self):
        """
        evd.equal_aspect()

        Set the aspect ratio of the plot so that x and y (anode dimensions)
        are visually equal in the axis.

        Parameters
        ----------
        None

        """
        extents = np.array([getattr(self.ax, 'get_{}lim'.format(dim))() for dim in 'xy'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xy'):
            getattr(self.ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
            
    def show(self):
        """
        evd.show()

        Display the plot, if there is a graphical backend and an attached display.

        Parameters
        ----------
        None

        """
        self.equal_aspect()
        plt.show()
        
    def save(self, outfile, **kwargs):
        """
        evd.save(outfile, **kwargs)

        Save the plot to disk.

        Parameters
        ----------
        outfile : str or os.path-like
            Location on disk to write the output file.

        Notes
        -----
        Additional kwargs are passed through to self.fig.savefig

        """
        self.equal_aspect()
        self.fig.savefig(outfile, **kwargs)

    def draw_box(self, cell_center_xy, cell_center_z, cell_pitch, cell_height, **kwargs):
        """
        evd.draw_box(cell_center_xy,
                     cell_center_z,
                     cell_pitch,
                     cell_height,
                     **kwargs)

        Draw a 3D box on the axes.

        Parameters
        ----------
        cell_center_xy : tuple(float, float)
            Position of box center in x and y.
        cell_center_z : float
            Position of box center in z.
        cell_pitch : float
            Extent of box in x and y directions.
        cell_height : float
            Extent of box in z direction.

        Notes
        -----
        Additional kwargs are passed through to Poly3DCollection

        """
        x_bounds = [cell_center_xy[0] - 0.5*cell_pitch,
                    cell_center_xy[0] + 0.5*cell_pitch]
        y_bounds = [cell_center_xy[1] - 0.5*cell_pitch,
                    cell_center_xy[1] + 0.5*cell_pitch]
        z_bounds = [cell_center_z,
                    cell_center_z + cell_height]
            
        bottom_face = np.array([[x_bounds[0], x_bounds[0],
                                 x_bounds[1], x_bounds[1], x_bounds[0]],
                                [y_bounds[0], y_bounds[1],
                                 y_bounds[1], y_bounds[0], y_bounds[0]],
                                [z_bounds[0], z_bounds[0],
                                 z_bounds[0], z_bounds[0], z_bounds[0]]]).T
        top_face = np.array([[x_bounds[0], x_bounds[0],
                              x_bounds[1], x_bounds[1], x_bounds[0]],
                             [y_bounds[0], y_bounds[1],
                              y_bounds[1], y_bounds[0], y_bounds[0]],
                             [z_bounds[1], z_bounds[1],
                              z_bounds[1], z_bounds[1], z_bounds[1]]]).T
        left_face = np.array([[x_bounds[0], x_bounds[0],
                               x_bounds[0], x_bounds[0], x_bounds[0]],
                              [y_bounds[0], y_bounds[1],
                               y_bounds[1], y_bounds[0], y_bounds[0]],
                              [z_bounds[0], z_bounds[0],
                               z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        right_face = np.array([[x_bounds[1], x_bounds[1],
                                x_bounds[1], x_bounds[1], x_bounds[1]],
                               [y_bounds[0], y_bounds[1],
                                y_bounds[1], y_bounds[0], y_bounds[0]],
                               [z_bounds[0], z_bounds[0],
                                z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        back_face = np.array([[x_bounds[0], x_bounds[1],
                               x_bounds[1], x_bounds[0], x_bounds[0]],
                              [y_bounds[0], y_bounds[0],
                               y_bounds[0], y_bounds[0], y_bounds[0]],
                              [z_bounds[0], z_bounds[0],
                               z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        front_face = np.array([[x_bounds[0], x_bounds[1],
                                x_bounds[1], x_bounds[0], x_bounds[0]],
                               [y_bounds[1], y_bounds[1],
                                y_bounds[1], y_bounds[1], y_bounds[1]],
                               [z_bounds[0], z_bounds[0],
                                z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        faces = [bottom_face,
                 top_face,
                 left_face,
                 right_face,
                 back_face,
                 front_face,
                 ]
        
        self.ax.add_collection3d(Poly3DCollection(faces, **kwargs))            

    def plot_raw_track(self, **kwargs):
        """
        evd.plot_raw_track(**kwargs)

        Plot the point sample representation of the undrifted track.

        Parameters
        ----------
        None

        Notes
        -----
        Additional kwargs are passed through to self.ax.scatter

        """

        n_points = self.track_object.raw_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.raw_track['position'][::reduction_factor,0]
            ys = self.track_object.raw_track['position'][::reduction_factor,1]
            zs = self.track_object.raw_track['position'][::reduction_factor,2]
            colors = np.log(self.track_object.raw_track['charge'][::reduction_factor])
        else:
            xs = self.track_object.raw_track['position'][:,0],
            ys = self.track_object.raw_track['position'][:,1],
            zs = self.track_object.raw_track['position'][:,2],
            colors = np.log(self.track_object.raw_track['charge'][:])
            
        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')
            
    def plot_drifted_track(self, **kwargs):
        """
        evd.plot_drifted_track(**kwargs)

        Plot the point sample representation of the drifted track.

        Parameters
        ----------
        None

        Notes
        -----
        Additional kwargs are passed through to self.ax.scatter

        """

        n_points = self.track_object.drifted_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.drifted_track['position'][::reduction_factor,0].cpu()
            ys = self.track_object.drifted_track['position'][::reduction_factor,1].cpu()
            zs = self.track_object.drifted_track['position'][::reduction_factor,2].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][::reduction_factor].cpu())
        else:
            xs = self.track_object.drifted_track['position'][:,0].cpu()
            ys = self.track_object.drifted_track['position'][:,1].cpu()
            zs = self.track_object.drifted_track['position'][:,2].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][:].cpu())
            
        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_drifted_track_timeline(self, **kwargs):
        """
        evd.plot_drifted_track_timeline(**kwargs)

        Plot the point sample representation of the drifted track. This method
        plots the arrival time on the z-axis instead of the true depth.

        Parameters
        ----------
        None

        Notes
        -----
        Additional kwargs are passed through to self.ax.scatter

        """

        n_points = self.track_object.drifted_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.drifted_track['position'][::reduction_factor,0].cpu()
            ys = self.track_object.drifted_track['position'][::reduction_factor,1].cpu()
            zs = self.track_object.drifted_track['time'][::reduction_factor].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][::reduction_factor].cpu())
        else:
            xs = self.track_object.drifted_track['position'][:,0].cpu()
            ys = self.track_object.drifted_track['position'][:,1].cpu()
            zs = self.track_object.drifted_track['time'][:].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][:].cpu())
                        

        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'Arrival Time [us]')
        
    def plot_coarse_tile_measurement(self, readout_config):
        """
        evd.plot_coarse_tile_measurement(readout_config)

        Plot the simulated coarse hits for an input track.

        Parameters
        ----------
        readout_config : ReadoutConfig object
            The readout config specifying the coarse tile pitch, clock
            interval, etc.  This should match the readout config used in
            the simulation step for this track.

        """

        for this_hit in self.track_object.coarse_tiles_samples:
            cell_center_xy = this_hit.coarse_cell_pos
            # cell_center_z = this_hit.coarse_measurement_time
            cell_center_t = this_hit.coarse_measurement_time
            v = 1.6e5
            cell_center_z = cell_center_t*v
            cell_hit_length = v*readout_config['coarse_tiles']['clock_interval']*readout_config['coarse_tiles']['integration_length']
            cell_measurement = this_hit.coarse_cell_measurement

            self.draw_box(cell_center_xy, cell_center_z,
                          readout_config['coarse_tiles']['pitch'],
                          cell_hit_length,
                          **self.coarse_tile_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_coarse_tile_measurement_timeline(self, readout_config):
        """
        evd.plot_coarse_tile_measurement_timeline(readout_config)

        Plot the simulated coarse hits for an input track. This method
        plots the arrival time on the z-axis instead of the true depth.

        Parameters
        ----------
        readout_config : ReadoutConfig object
            The readout config specifying the coarse tile pitch, clock
            interval, etc.  This should match the readout config used in
            the simulation step for this track.

        """

        xlim = []
        ylim = []
        zlim = []
        
        for this_hit in self.track_object.coarse_tiles_samples:
            cell_center_xy = this_hit.coarse_cell_pos
            # cell_center_z = this_hit.coarse_measurement_time
            cell_trigger_t = this_hit.coarse_measurement_time
            cell_measurement = this_hit.coarse_cell_measurement
            cell_hit_length = readout_config['coarse_tiles']['clock_interval']*readout_config['coarse_tiles']['integration_length']

            self.draw_box(cell_center_xy, cell_trigger_t,
                          readout_config['coarse_tiles']['pitch'],
                          cell_hit_length,
                          **self.coarse_tile_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'Arrival Time [us]')
        
    def plot_pixel_measurement(self, readout_config):
        """
        evd.plot_pixel_measurement(readout_config)

        Plot the simulated pixel hits for an input track.

        Parameters
        ----------
        readout_config : ReadoutConfig object
            The readout config specifying the coarse tile pitch, clock
            interval, etc.  This should match the readout config used in
            the simulation step for this track.

        """
        
        for this_hit in self.track_object.pixel_samples:
            cell_center_xy = this_hit.pixel_pos
            cell_center_t = this_hit.hit_timestamp
            v = 1.6e5
            cell_center_z = cell_center_t*v
            cell_measurement = this_hit.hit_measurement
            cell_hit_length = v*readout_config['pixels']['clock_interval']*readout_config['pixels']['integration_length']
            
            self.draw_box(cell_center_xy, cell_center_z,
                          readout_config['pixels']['pitch'],
                          cell_hit_length,
                          **self.pixel_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_pixel_measurement_timeline(self, readout_config):
        """
        evd.plot_pixel_measurement_timeline(readout_config)

        Plot the simulated pixel hits for an input track. This method
        plots the arrival time on the z-axis instead of the true depth.

        Parameters
        ----------
        readout_config : ReadoutConfig object
            The readout config specifying the coarse tile pitch, clock
            interval, etc.  This should match the readout config used in
            the simulation step for this track.

        """
        
        for this_hit in self.track_object.pixel_samples:
            cell_center_xy = this_hit.pixel_pos
            # cell_center_z = this_hit.hit_timestamp
            cell_trigger_t = this_hit.hit_timestamp
            cell_measurement = this_hit.hit_measurement
            cell_hit_length = readout_config['pixels']['clock_interval']*readout_config['pixels']['integration_length']
            
            self.draw_box(cell_center_xy, cell_trigger_t,
                          readout_config['pixels']['pitch'],
                          cell_hit_length,
                          **self.pixel_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'Arrival Time [us]')
