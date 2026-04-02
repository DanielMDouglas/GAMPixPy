from gampixpy.coordinates import CoordinateManager
from gampixpy.config import default_config_manager

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import SLACplots


coarse_tile_hit_kwargs = dict(facecolors=SLACplots.stanford.palo_verde,
                              linewidths=1,
                              edgecolors=SLACplots.SLACblue,
                              alpha = 0.)
pixel_hit_kwargs = dict(facecolors=SLACplots.stanford.illuminating,
                        linewidths=1,
                        edgecolors=SLACplots.stanford.illuminating,
                        alpha = 0.0)
TPC_boundary_kwargs = dict(facecolors=None,
                           linewidths=1,
                           edgecolors=SLACplots.stanford.full_palette['Black']['50%'],
                           linestyle = '--',
                           alpha = 0)

def draw_box(ax,
             cell_center_xy,
             cell_center_z,
             cell_pitch,
             cell_height,
             **kwargs):
    """
    draw_box(ax,
             cell_center_xy,
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

    corners = [[x_bounds[0], y_bounds[0], z_bounds[0]],
               [x_bounds[0], y_bounds[1], z_bounds[0]],
               [x_bounds[1], y_bounds[0], z_bounds[0]],
               [x_bounds[1], y_bounds[1], z_bounds[0]],
               [x_bounds[0], y_bounds[0], z_bounds[1]],
               [x_bounds[0], y_bounds[1], z_bounds[1]],
               [x_bounds[1], y_bounds[0], z_bounds[1]],
               [x_bounds[1], y_bounds[1], z_bounds[1]],
               ]

    draw_box_from_corners(ax, corners, **kwargs)

def draw_box_from_corners(ax,
                          corners,
                          **kwargs):
    """
    draw_box_from_bounds(ax,
                         corners,
                         **kwargs)
    
    Draw a 3D box on the axes.
    
    Parameters
    ----------
    corners : list(array)
        A list of the eight corners of the box to be drawn.

    Notes
    -----
    Additional kwargs are passed through to Poly3DCollection

    """
    bottom_face = np.array([corners[0],
                            corners[1],
                            corners[3],
                            corners[2],
                            corners[0],
                            ])
    top_face = np.array([corners[4],
                         corners[5],
                         corners[7],
                         corners[6],
                         corners[4],
                         ])
    left_face = np.array([corners[0],
                          corners[1],
                          corners[5],
                          corners[4],
                          corners[0],
                          ])
    right_face = np.array([corners[2],
                           corners[3],
                           corners[7],
                           corners[6],
                           corners[2],
                           ])
    front_face = np.array([corners[0],
                           corners[2],
                           corners[6],
                           corners[4],
                           corners[0],
                           ])
    back_face = np.array([corners[1],
                          corners[3],
                          corners[7],
                          corners[5],
                          corners[1],
                          ])
    faces = [bottom_face,
             top_face,
             left_face,
             right_face,
             back_face,
             front_face,
             ]

    ax.add_collection3d(Poly3DCollection(faces, **kwargs))            

def plot_tile_record(ax,
                    this_tile_record,
                    coordinate_manager,
                    config_manager,
                    z_offset = 0):
    tile_tpc = this_tile_record.tile_tpc
            
    tile_center_xy = this_tile_record.tile_pos
    tile_center_z = this_tile_record.trigger_depth

    tpc_coords = torch.tensor([tile_center_xy[0],
                               tile_center_xy[1],
                               tile_center_z])
    exp_coords = coordinate_manager.to_experiment_coords(tpc_coords,
                                                         tile_tpc)[0]
    exp_coords = exp_coords.cpu().numpy()
            
    cell_measurement = this_tile_record.waveform

    tile_config = config_manager.readout_config['coarse_tiles']
    
    pitch = tile_config['pitch'],
    
    this_volume_dict = config_manager.detector_config['drift_volumes'][coordinate_manager.index_to_volume[tile_tpc]]
    horizontal_axis = this_volume_dict['anode_horizontal'].cpu().numpy()
    half_span_horizontal = horizontal_axis*pitch/2

    vertical_axis = this_volume_dict['anode_vertical'].cpu().numpy()
    half_span_vertical = vertical_axis*pitch/2
    
    drift_axis = this_volume_dict['drift_axis'].cpu().numpy()
    v = config_manager.physics_config['charge_drift']['drift_speed']
    cell_hit_length = v*tile_config['clock_interval']*tile_config['integration_length']
    depth_span = drift_axis*cell_hit_length
    
    corners = [exp_coords - half_span_horizontal - half_span_vertical,
               exp_coords - half_span_horizontal + half_span_vertical,
               exp_coords + half_span_horizontal - half_span_vertical,
               exp_coords + half_span_horizontal + half_span_vertical,
               exp_coords - half_span_horizontal - half_span_vertical + depth_span,
               exp_coords - half_span_horizontal + half_span_vertical + depth_span,
               exp_coords + half_span_horizontal - half_span_vertical + depth_span,
               exp_coords + half_span_horizontal + half_span_vertical + depth_span,
               ]

    draw_box_from_corners(ax,
                          corners,
                          **coarse_tile_hit_kwargs)
    
def plot_pixel_record(ax,
                   this_pixel_record,
                   coordinate_manager,
                   config_manager,
                   z_offset = 0):
    pixel_tpc = this_pixel_record.pixel_tpc

    pixel_center_xy = this_pixel_record.pixel_pos
    pixel_z_series = this_pixel_record.trigger_depth

    for pixel_center_z in pixel_z_series:
        tpc_coords = torch.tensor([pixel_center_xy[0],
                                   pixel_center_xy[1],
                                   pixel_center_z + z_offset])

        exp_coords = coordinate_manager.to_experiment_coords(tpc_coords,
                                                             pixel_tpc)[0]
        exp_coords = exp_coords.cpu().numpy()

        pixel_config = config_manager.readout_config['pixels']
        pitch = pixel_config['pitch']
    
        this_volume_dict = config_manager.detector_config['drift_volumes'][coordinate_manager.index_to_volume[pixel_tpc]]
        horizontal_axis = this_volume_dict['anode_horizontal'].cpu().numpy()
        half_span_horizontal = horizontal_axis*pitch/2
    
        vertical_axis = this_volume_dict['anode_vertical'].cpu().numpy()
        half_span_vertical = vertical_axis*pitch/2
    
        drift_axis = this_volume_dict['drift_axis'].cpu().numpy()
        v = config_manager.physics_config['charge_drift']['drift_speed']
        pixel_hit_length = v*pixel_config['clock_interval']
        depth_span = drift_axis*pixel_hit_length

        corners = [exp_coords - half_span_horizontal - half_span_vertical,
                   exp_coords - half_span_horizontal + half_span_vertical,
                   exp_coords + half_span_horizontal - half_span_vertical,
                   exp_coords + half_span_horizontal + half_span_vertical,
                   exp_coords - half_span_horizontal - half_span_vertical + depth_span,
                   exp_coords - half_span_horizontal + half_span_vertical + depth_span,
                   exp_coords + half_span_horizontal - half_span_vertical + depth_span,
                   exp_coords + half_span_horizontal + half_span_vertical + depth_span,
                   ]

        draw_box_from_corners(ax, corners,
                              **pixel_hit_kwargs)

def plot_drift_volumes(ax, config_manager):
    volumes_dict = config_manager.detector_config['drift_volumes']
    for volume_name, volume_dict in volumes_dict.items():
        corners = [corner.cpu() for corner in 
                   volume_dict['anode_corners'] + volume_dict['cathode_corners']]
        draw_box_from_corners(ax, corners, **TPC_boundary_kwargs)
        
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

    Examples
    --------
    >>> evd = EventDisplay(event_data)
    >>> evd.plot_drifted_track_timeline(alpha = 0) # can also pass kwargs to plt.scatter
    >>> evd.plot_coarse_tile_measurement_timeline(readout_config) # plot tile hits
    >>> evd.plot_pixel_measurement_timeline(readout_config) # plot pixel hits
    >>> evd.show()

    """
    MAX_POINTS_PLOTTED = 1e3
    def __init__(self, track,
                 config_manager = default_config_manager):

        self.config_manager = config_manager

        self.track_object = track

        self._init_fig()
        
    def _init_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        # self.equal_aspect()

    def set_focus(self, string):
        """
        evd.set_focus(string)

        Sets the axis limits to contain a specific object.
        Options are:
        - 'coarse' : coarse tile hits
        - 'pixels' : pixel hits
        - 'charge' : ionized charge
        - 'tpc' : detector boundaries

        Parameers
        ---------
        string : str
            String specifying which object focus
        """

        # NOT IMPLEMENTED YET
        if string == 'coarse':
            pass
        elif string == 'pixels':
            pass
        elif string == 'charge':
            pass
        elif string == 'tpc':
            pass
        else:
            pass
        
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
        self.ax.axis('equal')
            
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

    def set_label_axes(self):
        """
        evd.set_label_axes()

        Add labels to axes.  These are the simple (x, y, z) and units (cm).
        
        """

        self.ax.set_xlabel(r'x [cm]')
        self.ax.set_ylabel(r'y [cm]')
        self.ax.set_zlabel(r'z [cm]')

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

        self.set_label_axes()

    def plot_drifted_track(self, **kwargs):
        """
        evd.plot_drifted_track(**kwargs)

        Plot the point sample representation of the drifted track.

        Notes
        -----
        Additional kwargs are passed through to self.ax.scatter

        """

        n_points = self.track_object.drifted_track['position'].shape[0]
        coordinate_manager = CoordinateManager(self.config_manager)
        coords = coordinate_manager.to_experiment_coords(self.track_object.drifted_track['position'],
                                                         self.track_object.tpc_track['TPC_index'])
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = coords[::reduction_factor,0].cpu()
            ys = coords[::reduction_factor,1].cpu()
            zs = coords[::reduction_factor,2].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][::reduction_factor].cpu())
        else:
            xs = coords[:,0].cpu()
            ys = coords[:,1].cpu()
            zs = coords[:,2].cpu()
            colors = np.log(self.track_object.drifted_track['charge'][:].cpu())
            
        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.set_label_axes()

    def plot_coarse_tile_measurement(self):
        """
        evd.plot_coarse_tile_measurement(readout_config)

        Plot the simulated coarse hits for an input track.
        
        """

        coordinate_manager = CoordinateManager(self.config_manager)
        
        for this_tile_record in self.track_object.coarse_tiles_samples:
            plot_tile_record(self.ax,
                             this_tile_record,
                             coordinate_manager,
                             self.config_manager,
                             )
            
        self.set_label_axes()
        
    def plot_pixel_measurement(self):
        """
        evd.plot_pixel_measurement(readout_config)

        Plot the simulated pixel hits for an input track.

        """

        coordinate_manager = CoordinateManager(self.config_manager)

        for this_pixel_record in self.track_object.pixel_samples:
            plot_pixel_record(self.ax,
                              this_pixel_record,
                              coordinate_manager,
                              self.config_manager,
                              )
            
        self.set_label_axes()

    def plot_drift_volumes(self):
        """
        evd.plot_drift_volumes(detector_config)

        Plot the boundaries of the drift volumes as specified in a
        detector configuration file.

        """

        plot_drift_volumes(self.ax,
                           self.config_manager.detector_config)

        self.set_label_axes()
