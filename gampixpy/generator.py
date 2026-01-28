import numpy as np
import torch

from gampixpy.tracks import Track
from gampixpy.input_parsing import meta_dtype

class Generator:
    """
    Generator(*args, **kwargs)

    Parent class for generator objects.  These are designed to work
    similarly to InputParser objects, i.e. present an iterator for
    event data and metadata arrays.

    Parameters
    ----------
    None

    Notes
    -----
    All args and kwargs to this class are saved as attributes, to
    generalize inputs for subclasses.

    See Also
    --------
    PointSource : Sub-class for generating tracks with a point profile.
    LineSource : Sub-class for generating line-like tracks with a
                 uniform charge density.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        while True:
            yield self.get_sample(), self.get_meta()
    
class PointSource (Generator):
    """
    PointSource(xrange=None,
                yrange=None,
                zrange=None,
                trange=None,
                qrange=None)

    A generator for point sources uniformly distributed within a
    rectangular volume.  The keyword arguments specify the boundaries
    of the region from which a point is randomly sampled, and a track
    object with a preset number of points (1e5) is generated.

    Attributes
    ----------
    xrange : tuple(float, float)
        Range of x values from which to sample
    yrange : tuple(float, float)
        Range of y values from which to sample
    zrange : tuple(float, float)
        Range of z values from which to sample
    trange : tuple(float, float)
        Range of t values from which to sample
    qrange : tuple(float, float)
        Range of q values from which to sample

    Examples
    --------
    >>> ps_gen = PointSource(x_range=(0, 1),
                             y_range=(0, 1),
                             z_range=(0, 1),
                             t_range=(0, 1),
                             q_range=(0, 1))
    >>> point_cloud = ps_gen.get_sample()
    >>> meta = ps_gen.get_meta()
    >>> detector_model.simulate(point_cloud)
    
    See Also
    --------
    Generator : Parent class for generators.
    LineSource : Sub-class for generating line-like tracks with a
                 uniform charge density.
    
    """
    def generate_sample_params(self):
        """
        gen.generate_sample_params()

        Draw a new sample from the distribution parameters.

        Parameters
        ----------
        None
        
        """
        self.n_samples_per_point = 100000

        self.x_init = self.kwargs['x_range'][0] + (self.kwargs['x_range'][1] - self.kwargs['x_range'][0])*np.random.random()
        self.y_init = self.kwargs['y_range'][0] + (self.kwargs['y_range'][1] - self.kwargs['y_range'][0])*np.random.random()
        self.z_init = self.kwargs['z_range'][0] + (self.kwargs['z_range'][1] - self.kwargs['z_range'][0])*np.random.random()
        self.t_init = self.kwargs['t_range'][0] + (self.kwargs['t_range'][1] - self.kwargs['t_range'][0])*np.random.random()
        self.q_init = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

    def get_sample(self):
        """
        gen.get_sample()

        Generate a new sample from the parametrized distribution and return
        a Track object.
        
        Parameters
        ----------
        None

        Returns
        -------
        track : Track object
            An instance of the Track class where the sample points lie on the
            same point described in the sample parameters with uniform charge
            density.
        
        """
        
        self.generate_sample_params()

        charge_position = torch.tensor(self.n_samples_per_point*[[self.x_init,
                                                                  self.y_init,
                                                                  self.z_init,
                                                                  ]])
        charge_time = torch.tensor(self.n_samples_per_point*[self.t_init,
                                                             ])

        charge_values = torch.tensor(self.n_samples_per_point*[self.q_init/self.n_samples_per_point,
                                                               ])
        sample_labels = torch.zeros(self.n_samples_per_point)

        return Track(charge_position,
                     charge_time,
                     charge_values,
                     sample_labels)

    def get_meta(self):
        """
        gen.get_meta()

        Return the metadata associated with the current set of
        sample parameters.

        Parameters
        ----------
        None

        Returns
        -------
        meta_array : numpy.array
            An array containing metadata for this sample.  The
            dtype for this array is:

                0    "event id"            "u4"    (undefined)
                1    "primary energy"      "f4"    (undefined)
                2    "deposited charge"    "f4"
                3    "vertex x"            "f4"
                4    "vertex y"            "f4"
                5    "vertex z"            "f4"
                6    "theta"               "f4"    (undefined)
                7    "phi"                 "f4"    (undefined)
                8    "primary length"      "f4"    (undefined)

            as described in gampixpy.input_parsing.meta_dtype

        """
        
        meta_array = np.array([(0, 0,
                                self.q_init,
                                self.x_init,
                                self.y_init,
                                self.z_init,
                                0,
                                0,
                                0)],
                              dtype = meta_dtype)
                
        return meta_array 

class LineSource (Generator):
    """
    LineSource(xrange=None,
               yrange=None,
               zrange=None,
               trange=None,
               qrange=None,
               lenght_range=None)

    A generator for line sources uniformly distributed within a
    rectangular volume.  The keyword arguments specify the boundaries
    of the region from which a point is randomly sampled, and a track
    object with a preset number of points (1e5) is generated.

    Attributes
    ----------
    xrange : tuple(float, float)
        Range of x values from which to sample
    yrange : tuple(float, float)
        Range of y values from which to sample
    zrange : tuple(float, float)
        Range of z values from which to sample
    trange : tuple(float, float)
        Range of t values from which to sample
    qrange : tuple(float, float)
        Range of q values from which to sample
    length_range : tuple(float, float)
        Range of tracklet length values from which to sample

    Examples
    --------
    >>> ls_gen = LineSource(x_range=(0, 1),
                            y_range=(0, 1),
                            z_range=(0, 1),
                            t_range=(0, 1),
                            q_range=(0, 1),
                            length_range=(1,2))
    >>> line_cloud = ls_gen.get_sample()
    >>> meta = ls_gen.get_meta()
    >>> detector_model.simulate(line_cloud)
    
    See Also
    --------
    Generator : Parent class for generators.
    PointSource : Sub-class for generating tracks with a point profile.
    
    """

    def generate_sample_params(self):
        """
        gen.generate_sample_params()

        Sample from the distribution parameters.

        Parameters
        ----------
        None
        
        """
        self.n_samples = 10000

        self.x_init = self.kwargs['x_range'][0] + (self.kwargs['x_range'][1] - self.kwargs['x_range'][0])*np.random.random()
        self.y_init = self.kwargs['y_range'][0] + (self.kwargs['y_range'][1] - self.kwargs['y_range'][0])*np.random.random()
        self.z_init = self.kwargs['z_range'][0] + (self.kwargs['z_range'][1] - self.kwargs['z_range'][0])*np.random.random()
        self.t_init = self.kwargs['t_range'][0] + (self.kwargs['t_range'][1] - self.kwargs['t_range'][0])*np.random.random()
        self.q = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

        # maybe direction parameters can be passed as kwargs?
        # for now, let's just always use a uniform distribution
        # over the unit sphere
        self.theta = 2*torch.pi*torch.rand(1)
        self.phi = torch.arccos(1 - 2*torch.rand(1))

        self.length = self.kwargs['length_range'][0] + (self.kwargs['length_range'][1] - self.kwargs['length_range'][0])*torch.rand(1)

    def do_point_sampling(self, start_4vec, end_4vec,
                          dx, charge_per_sample):
        """
        gen.do_point_sampling(start_4vec, end_4vec,
                              dx, charge_per_sample)

        Sample points with uniform density along a track described
        by two endpoints.

        Parameters
        ----------
        start_4vec : array-like[float, float, float, float]
            (position, time) vector for the initial point of the line
            segment.
        end_4vec : array-like[float, float, float, float]
            (position, time) vector for the final point of the line
            segment.
        dx : float
            Length of the line segment.
        charge_per_sample : float
            Charge on each point sample within the line distribution.
            Should be equivalent to total charge/n_samples

        Returns
        -------
        sample_position : array-like[float, float, float]
            Position vector for each point within the line distribution.
        sample_time : array-like[float, float, float, float]
            Time for each point within the line distribution.
        sample_charges : array-like[float]
            charge per point sample
        
        """
        # point sampling with a fixed number of samples per length
        # it may be faster to do sampling another way (test in future!)
        #  - sample with fixed amount of charge
        #  - sample with fixed number of samples per segment

        segment_interval = end_4vec - start_4vec

        sample_parametric_distance = torch.linspace(0, 1, self.n_samples)
        sample_4vec = start_4vec + segment_interval*sample_parametric_distance[:,None]

        sample_position = sample_4vec[:,:3]
        sample_time = sample_4vec[:,3]
        sample_charges = charge_per_sample*torch.ones(self.n_samples)

        return sample_position, sample_time, sample_charges
        
    def get_sample(self):
        """
        gen.get_sample()

        Generate a new sample from the parametrized distribution and return
        a Track object.
        
        Parameters
        ----------
        None

        Returns
        -------
        track : Track object
            An instance of the Track class where the sample points lie along
            the line described in the sample parameters with uniform charge
            density.
        
        """
        self.generate_sample_params()

        start_4vec = torch.tensor((self.x_init,
                                   self.y_init,
                                   self.z_init,
                                   self.t_init,
                                   ))
        dir_4vec = torch.tensor([torch.cos(torch.tensor(self.theta))*torch.sin(torch.tensor(self.phi)),
                                 torch.sin(torch.tensor(self.theta))*torch.sin(torch.tensor(self.phi)),
                                 torch.cos(torch.tensor(self.phi)),
                                 0,
                                 ])
        end_4vec = start_4vec + dir_4vec*self.length

        displacement = start_4vec[:3] - end_4vec[:3]
        dx = torch.sum(displacement**2)
        dQ = self.q/self.n_samples
        charge_4vec, charge_values = self.do_point_sampling(start_4vec,
                                                            end_4vec,
                                                            dx, dQ,
                                                            )

        return Track(charge_4vec, charge_values)

    def get_meta(self):
        """
        gen.get_meta()

        Return the metadata associated with the current set of
        sample parameters.

        Parameters
        ----------
        None

        Returns
        -------
        meta_array : numpy.array
            An array containing metadata for this sample.  The
            dtype for this array is:

                0    "event id"            "u4"    (undefined)
                1    "primary energy"      "f4"    (undefined)
                2    "deposited charge"    "f4"
                3    "vertex x"            "f4"
                4    "vertex y"            "f4"
                5    "vertex z"            "f4"
                6    "theta"               "f4"
                7    "phi"                 "f4"
                8    "primary length"      "f4"

            as described in gampixpy.input_parsing.meta_dtype

        """

        meta_array = np.array([(0, 0,
                                self.q,
                                self.x_init,
                                self.y_init,
                                self.z_init,
                                self.theta,
                                self.phi,
                                self.length)],
                              dtype = meta_dtype)
                
        return meta_array 
