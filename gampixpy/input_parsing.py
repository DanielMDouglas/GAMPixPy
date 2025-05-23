from gampixpy.tracks import Track
from gampixpy.config import default_physics_params
from gampixpy.units import *
from gampixpy.recombination import BoxRecombinationModel, BirksRecombinationModel

import numpy as np
import torch
import particle

meta_dtype =  np.dtype([("event id", "u4"),
                        ("primary energy", "f4"),
                        ("deposited charge", "f4"),
                        ("vertex x", "f4"),
                        ("vertex y", "f4"),
                        ("vertex z", "f4"),
                        ("theta", "f4"),
                        ("phi", "f4"),
                        ("primary length", "f4"),
                        ],
                       align = True)

class InputParser:
    """
    InputParser(input_filename,
                sequential_sampling = True,
                physics_config = default_physics_params)

    Parent class for more specialized input parsers.

    Attributes
    ----------
    physics_config : PhysicsConfig object
        Physics configuration.  Some early physics processes are handled
        by the input parser (for now!), such as charge/light yield
        calculation.
    input_filename : str or os.path-like
        Path to the input data on disk.
    sampling_order : array-like
        Array describing the order for iterating through the input indices.
    """
    def __init__(self, input_filename,
                 position_offset = [0, 0, 0],
                 sequential_sampling = True,
                 physics_config = default_physics_params):
        self.physics_config = physics_config
        
        self.input_filename = input_filename

        self.sampling_order = []
        self.global_position_offset = torch.tensor(position_offset)

        self._open_file_handle()
        self._generate_sample_order(sequential_sampling)

    def __len__(self):
        if 'n_images' in dir(self):
            return self.n_images
        else:
            return None
        
    def __iter__(self):
        for sample_index in self.sampling_order:
            yield sample_index.item(), self.get_sample(sample_index.item()), self.get_meta(sample_index.item())

class SegmentParser (InputParser):
    """
    SegmentParser(input_filename,
                sequential_sampling = True,
                physics_config = default_physics_params)

    Parent class for segment-based input parsers.  This class defines
    methods for point sampline along segment-like inputs and recombination.

    Attributes
    ----------
    physics_config : PhysicsConfig object
        Physics configuration.  Some early physics processes are handled
        by the input parser (for now!), such as charge/light yield
        calculation.
    input_filename : str or os.path-like
        Path to the input data on disk.
    sampling_order : array-like
        Array describing the order for iterating through the input indices.
    
    """
    def do_recombination(self, dE, dx, dEdx, mode = 'birks', **kwargs):
        """
        parser.do_recombination(dE, dx, dEdx, mode = 'birks', **kwargs)

        Calculate the recombination factor from input segment attributes.

        Parameters
        ----------
        dE : array-like[float]
            Energy deposition (integrated) over the input segment.
        dx : array-like[float]
            Length of each input segment.
        dEdx : array-like[float]
            Energy-deposition density (average) over the input segment.
            
        mode : string
            Recombination model to choose from.  Currently implemented are
            'box' and 'birks'.

        Returns
        -------
        n_electrons : array-like[float]
            Charge yield in number of electrons for each segment described
            in the input arrays.

        See Also
        --------
        recombination.BoxRecombinationModel : Implementation of the Box
            recombination model.
        recombination.BirksRecombinationModel : Implementation of the Birks
            recombination model.
        
        """
        if mode.lower() == 'box':
            recombination_model = BoxRecombinationModel(self.physics_config)
        elif mode.lower() == 'birks':
            recombination_model = BirksRecombinationModel(self.physics_config)
        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BRIKS'")

        recomb = recombination_model(dE, dx, dEdx)

        if torch.any(torch.isnan(recomb)):
            raise RuntimeError("Invalid recombination value")

        w_ion = self.physics_config['material']['w']
        charge_yield_per_energy = recomb/w_ion

        n_electrons = dE*charge_yield_per_energy
        return n_electrons
    
    def do_point_sampling(self, start_4vec, end_4vec,
                          dx, charge_per_segment,
                          sample_density = 1.e-1,
                          sample_normalization = 'charge',
                          **kwargs):
        """
        parser.do_point_sampling(start_4vec, end_4vec,
                                 dx, charge_per_segment,
                                 sample_density = 1.e-1,
                                 sample_normalization = 'charge',
                                 **kwargs)

        Sample points along the segments described by the input arrays.

        Parameters
        ----------
        start_4vec : array-like
            (position, time) 4-vector array for the initial points of
            the image segments.
        end_4vec : array-like
            (position, time) 4-vector array for the final points of
            the image segments.
        dx : array-like
            Length array for the image segments.
        charge_per_segment : array-like
            Charge per segment array for the image segments.
        sample_density : float
            Density of samples.  Exact interpretation depends upon the
            normalization method specified.
        sample_normalization : str
            Normalization method for choosing sample density:

                'charge' : Using this method, sample_density has units
                    of [samples/e].  The samples will therefore also
                    have charge e/sample_density.
                'length' : Using this method, sample_density has units
                    of [samples/cm].  This method has the potential to
                    ignore extremely small segments with high ionization.
        
        Returns
        -------
        sample_position : array-like[float, float, float]
            Interpolated position along the input segments.
        sample_time : array-like[float]
            Interpolated ionization time along the input segments.
        charges : array-like[float]
            Charge values for each interpolated point.
        
        """

        segment_interval = end_4vec - start_4vec

        if sample_normalization == 'charge':
            # here, sample_density is [samples/unit charge]
            samples_per_segment = (charge_per_segment*sample_density).int()
        elif sample_normalization == 'length':
            # here, sample_density is [samples/unit length]
            samples_per_segment = (dx*sample_density).int()

        sample_start = torch.repeat_interleave(start_4vec, samples_per_segment, dim = 0)
        sample_interval = torch.repeat_interleave(segment_interval, samples_per_segment, dim = 0)
        sample_parametric_distance = torch.cat(tuple(torch.linspace(0, 1, samples_per_segment[i])
                                                     for i in range(samples_per_segment.shape[0])))
        sample_4vec = sample_start + sample_interval*sample_parametric_distance[:,None]

        sample_position = sample_4vec[:,:3]
        sample_time = sample_4vec[:,3]
        sample_charge = torch.repeat_interleave(charge_per_segment/samples_per_segment, samples_per_segment)

        return sample_position, sample_time, sample_charge

class RooTrackerParser (SegmentParser):
    """
    RooTrackerParser(input_filename,
                     sequential_sampling = True,
                     physics_config = default_physics_params)

    Class for parsing inputs from ROOT (edep-sim output).

    Attributes
    ----------
    physics_config : PhysicsConfig object
        Physics configuration.  Some early physics processes are handled
        by the input parser (for now!), such as charge/light yield
        calculation.
    input_filename : str or os.path-like
        Path to the input data on disk.
    sampling_order : array-like
        Array describing the order for iterating through the input indices.
    
    """
    def _open_file_handle(self, **kwargs):
        from ROOT import TFile, TG4Event

        self.file_handle = TFile(self.input_filename)
        self.inputTree = self.file_handle.Get("EDepSimEvents")

        self.event = TG4Event()
        self.inputTree.SetBranchAddress("Event", self.event)

    def _generate_sample_order(self, sequential_sampling, **kwargs):
        self.n_images = self.inputTree.GetEntriesFast()
        if sequential_sampling:
            self.sampling_order = torch.arange(self.n_images)
        else:
            self.sampling_order = torch.randperm(self.n_images)

    def _get_G4_segments(self, sample_index, position_offset = None, **kwargs):
        self.inputTree.GetEntry(sample_index, **kwargs)

        traj_id = torch.empty((0))
        traj_pdg = torch.empty((0))
        for trajectory in self.event.Trajectories:
            this_traj_id = torch.tensor([trajectory.GetTrackId()])
            this_traj_pdg = torch.tensor([trajectory.GetPDGCode()])

            traj_id = torch.cat((this_traj_id,
                                 traj_id))
            traj_pdg = torch.cat((this_traj_pdg,
                                 traj_pdg))
        
        start_pos = torch.empty((0,4))
        end_pos = torch.empty((0,4))
        start_time = torch.empty((0))
        end_time = torch.empty((0))
        dE = torch.empty((0))
        pdgid = torch.empty((0))

        for container_name, hit_segments in self.event.SegmentDetectors:
            for segment in hit_segments:

                this_start_pos = torch.tensor([segment.GetStart().X()*mm,
                                                segment.GetStart().Y()*mm,
                                                segment.GetStart().Z()*mm])
                this_end_pos = torch.tensor([segment.GetStop().X()*mm,
                                              segment.GetStop().Y()*mm,
                                              segment.GetStop().Z()*mm])

                this_start_time = torch.tensor([segment.GetStart().T()*ns])
                this_end_time = torch.tensor([segment.GetStop().T()*ns])

                this_dE = torch.tensor([segment.GetEnergyDeposit()*MeV])

                parent_id = list(segment.Contrib)[0]
                this_pdgid = traj_pdg[traj_id == parent_id]
                
                start_pos = torch.cat((this_start_pos[None,:],
                                       start_pos))
                end_pos = torch.cat((this_end_pos[None,:],
                                      end_pos))
                start_time = torch.cat((this_start_time,
                                        start_time))
                end_time = torch.cat((this_end_time,
                                      end_time))
                dE = torch.cat((this_dE,
                                dE))
                pdgid = torch.cat((this_pdgid,
                                   pdgid))

        return start_pos, end_pos, start_time, end_time, dE, pdgid
            
    def _get_G4_sample(self, sample_index, position_offset = None, **kwargs):
        self.inputTree.GetEntry(sample_index, **kwargs)

        start_4vec = torch.empty((0,4))
        end_4vec = torch.empty((0,4))
        dE = torch.empty((0))

        for container_name, hit_segments in self.event.SegmentDetectors:
            for segment in hit_segments:

                this_start_4vec = torch.tensor([segment.GetStart().X()*mm,
                                                segment.GetStart().Y()*mm,
                                                segment.GetStart().Z()*mm,
                                                segment.GetStart().T()*ns])
                this_end_4vec = torch.tensor([segment.GetStop().X()*mm,
                                              segment.GetStop().Y()*mm,
                                              segment.GetStop().Z()*mm,
                                              segment.GetStop().T()*ns])
                this_dE = torch.tensor([segment.GetEnergyDeposit()*MeV])
                
                start_4vec = torch.cat((this_start_4vec[None,:],
                                        start_4vec))
                end_4vec = torch.cat((this_end_4vec[None,:],
                                      end_4vec))
                dE = torch.cat((this_dE,
                                dE))

        displacement = start_4vec[:,:3] - end_4vec[:,:3]
        dx = torch.sqrt(torch.sum(displacement**2, dim = 1))
        dEdx = torch.where(dx > 0, dE/dx, 0.)

        dQ = self.do_recombination(dE, dx, dEdx, **kwargs)
        charge_position, charge_time, charge_values = self.do_point_sampling(start_4vec,
                                                                             end_4vec,
                                                                             dx, dQ,
                                                                             **kwargs
                                                                             )

        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
            
        return Track(charge_position + offset,
                     charge_time,
                     charge_values)

    def _get_G4_meta(self, sample_index,
                     position_offset = None,
                     **kwargs):
        primary_vertex = self.event.Primaries[0] # assume only one primary for now

        offset = np.array(self.global_position_offset.cpu())
        if position_offset:
            offset += np.array(position_offset)

        vertex_x = primary_vertex.GetPosition().X()*mm + offset[0]
        vertex_y = primary_vertex.GetPosition().Y()*mm + offset[1]
        vertex_z = primary_vertex.GetPosition().Z()*mm + offset[2]
        
        primary_trajectory = self.event.Trajectories[0]
        assert primary_trajectory.GetParentId() == -1

        pdg_code = primary_trajectory.GetPDGCode()
        mass = particle.Particle.from_pdgid(pdg_code).mass*MeV
        energy = primary_trajectory.GetInitialMomentum()[3]*MeV
        momentum = [primary_trajectory.GetInitialMomentum()[i]*MeV for i in range(3)]
        kinetic_energy = energy - mass

        theta = np.arctan2(momentum[1], momentum[0])
        phi = np.arctan2(np.sqrt(momentum[0]**2 + momentum[1]**2), momentum[2])
        
        meta_array = np.array([(sample_index,
                                kinetic_energy,
                                -1, # charge undefined
                                vertex_x, vertex_y, vertex_z,
                                theta, phi,
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
        return meta_array

    def get_segments(self, index, **kwargs):
        """
        parser.get_segments(index, **kwargs)

        Get the N Geant4 segments associated with this event index from the
        input file.  These are normally passed directly to the point sampling
        method when `get_sample` is called.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        start_pos : array-like
            (N, 3) array containing the start points for each segment in the event.
        end_pos : array-like
            (N, 3) array containing the final points for each segment in the event.
        start_time : array-like
            (N,) array containing the time of the initial point in each segment in
            the event.
        end_time : array-like
            (N,) array containing the time of the final point in each segment in the
            event.
        dE : array-like
            (N,) array containing the energy deposited via ionizations for each
            segment in the event.
        pdgid : array-like
            (N,) array containing the PDG code for each segment in the event.
                
        """
        return self._get_G4_segments(index, **kwargs)

    def get_sample(self, index, **kwargs):
        """
        parser.get_sample(index, **kwargs)

        Get the sample image from the loaded file.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        sample : Track object
            Return the loaded image as a point-sampled track object.
        
        """
        return self._get_G4_sample(index, **kwargs)

    def get_meta(self, index, **kwargs):
        """
        parser.get_meta(index, **kwargs)

        Get the metadata from the loaded file

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        meta_array : array-like[float]
            Return an array containing the metadata for this sample.  The dtype
            of this array is defined in meta_dtype, above.
        
        """
        return self._get_G4_meta(index, **kwargs)

class EdepSimParser (SegmentParser):
    # Unit conventions for edepsim inputs:
    # distance: cm
    # energy: MeV
    def _open_file_handle(self, **kwargs):
        import h5py
        self.file_handle = h5py.File(self.input_filename, **kwargs)

    def _generate_sample_order(self, sequential_sampling, **kwargs):
        unique_event_ids = np.unique(self.file_handle['trajectories']['eventID']).astype(np.int32)
        # unique_event_ids = torch.tensor(unique_event_ids, dtype = torch.int32)
        self.n_images = len(unique_event_ids)
        if sequential_sampling:
            self.sampling_order = torch.tensor(unique_event_ids)
        else:
            self.sampling_order = torch.tensor(unique_event_ids[torch.randperm(self.n_images)])

    def _get_edepsim_segments(self, sample_index,
                              pdg_selection=None,
                              position_offset = None,
                              **kwargs):
        segment_mask = self.file_handle['segments']['eventID'] == sample_index
        if type(pdg_selection) in [np.ndarray, list]:
            pdg_selection = np.array(pdg_selection)
            segment_mask *= np.sum(self.file_handle['segments']['pdgId'] == pdg_selection[:,None],
                                   axis = 0, dtype = bool)
        elif pdg_selection:
            segment_mask *= self.file_handle['segments']['pdgId'] == pdg_selection
        event_segments = self.file_handle['segments'][segment_mask]
      
        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]

        start_pos = np.array([event_segments['x_start']*cm,
                              event_segments['y_start']*cm,
                              event_segments['z_start']*cm]).T
        start_pos = torch.tensor(start_pos)
        end_pos = np.array([event_segments['x_end']*cm,
                            event_segments['y_end']*cm,
                            event_segments['z_end']*cm]).T
        end_pos = torch.tensor(end_pos)
        
        start_time = torch.tensor(event_segments['t_start']*ns)
        end_time = torch.tensor(event_segments['t_end']*ns)
        
        dE = torch.tensor(event_segments['dE']*MeV)
        pdgid = torch.tensor(event_segments['pdgId'])

        return start_pos, end_pos, start_time, end_time, dE, pdgid
            
    def _get_edepsim_event(self, sample_index,
                           pdg_selection=None,
                           position_offset = None,
                           **kwargs):
        segment_mask = self.file_handle['segments']['eventID'] == sample_index
        if type(pdg_selection) in [np.ndarray, list]:
            pdg_selection = np.array(pdg_selection)
            segment_mask *= np.sum(self.file_handle['segments']['pdgId'] == pdg_selection[:,None],
                                   axis = 0, dtype = bool)
        elif pdg_selection:
            segment_mask *= self.file_handle['segments']['pdgId'] == pdg_selection
        event_segments = self.file_handle['segments'][segment_mask]
      
        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]

        start_4vec = np.array([event_segments['x_start']*cm,
                               event_segments['y_start']*cm,
                               event_segments['z_start']*cm,
                               event_segments['t_start']*ns,
                               ]).T
        start_4vec = torch.tensor(start_4vec)
        end_4vec = np.array([event_segments['x_end']*cm,
                             event_segments['y_end']*cm,
                             event_segments['z_end']*cm,
                             event_segments['t_end']*ns,
                             ]).T
        end_4vec = torch.tensor(end_4vec)
        dE = torch.tensor(event_segments['dE']*MeV)

        displacement = start_4vec[:,:3] - end_4vec[:,:3]
        dx = torch.sqrt(torch.sum(displacement**2, dim = 1))
        dEdx = torch.where(dx > 0, dE/dx, 0.)

        dQ = self.do_recombination(dE, dx, dEdx, **kwargs)
        charge_position, charge_time, charge_values = self.do_point_sampling(start_4vec,
                                                                             end_4vec,
                                                                             dx, dQ,
                                                                             **kwargs
                                                                             )
        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
            
        return Track(charge_position + offset,
                     charge_time,
                     charge_values)
    
    def _get_edepsim_meta(self, sample_index,
                          position_offset = None,
                          **kwargs):
        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]
        primary_trajectory = event_trajectories[event_trajectories['parentID'] == -1]

        pdg_code = primary_trajectory['pdgId']
        mass = particle.Particle.from_pdgid(pdg_code).mass # MeV/c^2
        momentum = primary_trajectory['pxyz_start'] # MeV/c
        kinetic_energy = np.sqrt(np.power(mass, 2) + np.sum(np.power(momentum, 2))) - mass

        offset = np.array(self.global_position_offset.cpu())
        if position_offset:
            offset += np.array(position_offset)

        vertex = primary_trajectory['xyz_start'][0] + offset
        init_momentum = primary_trajectory['pxyz_start'][0]

        theta = np.arctan2(init_momentum[1], init_momentum[0])
        phi = np.arctan2(np.sqrt(init_momentum[0]**2 + init_momentum[1]**2), init_momentum[2])
        
        meta_array = np.array([(sample_index,
                                kinetic_energy,
                                -1, # charge undefined
                                vertex[0], vertex[1], vertex[2],
                                theta, phi,
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
        return meta_array

    def get_segments(self, index, **kwargs):
        """
        parser.get_segments(index, **kwargs)

        Get the N Geant4 segments associated with this event index from the
        input file.  These are normally passed directly to the point sampling
        method when `get_sample` is called.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        start_pos : array-like
            (N, 3) array containing the start points for each segment in the event.
        end_pos : array-like
            (N, 3) array containing the final points for each segment in the event.
        start_time : array-like
            (N,) array containing the time of the initial point in each segment in
            the event.
        end_time : array-like
            (N,) array containing the time of the final point in each segment in the
            event.
        dE : array-like
            (N,) array containing the energy deposited via ionizations for each
            segment in the event.
        pdgid : array-like
            (N,) array containing the PDG code for each segment in the event.
                
        """
        return self._get_edepsim_segments(index, **kwargs)
        
    def get_sample(self, index, **kwargs):
        """
        parser.get_sample(index, **kwargs)

        Get the sample image from the loaded file.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        sample : Track object
            Return the loaded image as a point-sampled track object.
        
        """
        return self._get_edepsim_event(index, **kwargs)

    def get_meta(self, index, **kwargs):
        """
        parser.get_meta(index, **kwargs)

        Get the metadata from the loaded file

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        meta_array : array-like[float]
            Return an array containing the metadata for this sample.  The dtype
            of this array is defined in meta_dtype, above.
        
        """
        return self._get_edepsim_meta(index, **kwargs)

class MarleyParser (SegmentParser):
    # BROKEN
    # revisit this later
    def _open_file_handle(self):
        from ROOT import TFile

        self.file_handle = TFile(self.input_filename)
        self.inputTree = self.file_handle.Get("edep")

    def _generate_sample_order(self, sequential_sampling):
        event_ids = []
        for entry in self.inputTree:
            event_ids.append(entry.event)
        unique_event_ids = np.unique(event_ids)
        self.n_images = self.inputTree.GetEntriesFast()
        if sequential_sampling:
            self.sampling_order = unique_event_ids
        else:
            self.sampling_order = np.random.choice(unique_event_ids,
                                                   len(unique_event_ids),
                                                   replace = False)
    
    def _get_G4_sample(self, sample_index,
                       position_offset = None,
                       **kwargs,
                       ):
        segment_dtype = np.dtype([("x_start", "f4"),
                                  ("y_start", "f4"),
                                  ("z_start", "f4"),
                                  ("x_end", "f4"),
                                  ("y_end", "f4"),
                                  ("z_end", "f4"),
                                  ("dE", "f4"),
                                  ("dx", "f4"),
                                  ("dEdx", "f4")],
                                align = True)
        segment_array = np.array([], dtype = segment_dtype)

        for entry in self.inputTree:
            if entry.event == sample_index:
                x_start = entry.startX
                y_start = entry.startY
                z_start = entry.startZ
                x_end = entry.endX
                y_end = entry.endY
                z_end = entry.endZ

                x_d = x_end - x_start
                y_d = y_end - y_start
                z_d = z_end - z_start
                dx = np.sqrt(x_d**2 + y_d**2 + z_d**2)
                
                dE = entry.dE
                dEdx = dE/dx if dx > 0 else 0

                this_segment_array = np.array([(x_start,
                                                y_start,
                                                z_start,
                                                x_end,
                                                y_end,
                                                z_end,
                                                dE,
                                                dx,
                                                dEdx)],
                                              dtype = segment_dtype)
                segment_array = np.concatenate((this_segment_array,
                                                segment_array))

        charge_per_segment = self.do_recombination(segment_array)
        charge_position, charge_time, charge_values = self.do_point_sampling(segment_array, charge_per_segment)

        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
            
        return Track(charge_position + offset,
                     charge_time,
                     charge_values)

    def _get_G4_meta(self, sample_index,
                     **kwargs):
        meta_array = np.array([(sample_index,
                                -1, # KE undefined
                                -1, # charge undefined
                                -1, -1, -1, # vertex point undefined
                                -1, -1, # primary attitude undefined
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
        return meta_array

    def get_sample(self, index, **kwargs):
        """
        parser.get_sample(index, **kwargs)

        Get the sample image from the loaded file.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        sample : Track object
            Return the loaded image as a point-sampled track object.
        
        """
        return self.get_G4_sample(index, **kwargs)

    def get_meta(self, index, **kwargs):
        """
        parser.get_meta(index, **kwargs)

        Get the metadata from the loaded file

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        meta_array : array-like[float]
            Return an array containing the metadata for this sample.  The dtype
            of this array is defined in meta_dtype, above.
        
        """
        return self.get_G4_meta(index, **kwargs)

class MarleyCSVParser (SegmentParser):
    def _open_file_handle(self):
        self.col_names = ('run', 'subrun', 'event',
                          'isSignal', 'pdgCode', 'trackID',
                          'motherID', 'startE', 'dE',
                          'startX', 'startY',
                          'startZ', 'startT',
                          'endX', 'endY',
                          'endZ', 'endT',
                          )
        self.col_types = ('i4', 'i4', 'i4',
                          'b1', 'i4', 'i4',
                          'i4', 'f4', 'f4',
                          'f4', 'f4',
                          'f4', 'f4',
                          'f4', 'f4',
                          'f4', 'f4',
                          )

        self.data_table = np.loadtxt(self.input_filename,
                                     skiprows = 1, # first row is a header column
                                     delimiter = ',', 
                                     dtype = {'names': self.col_names,
                                              'formats': self.col_types},
                                     )

    def _generate_sample_order(self, sequential_sampling):

        unique_event_ids = np.unique(self.data_table['event']).astype(np.int32)
        if sequential_sampling:
            self.sampling_order = torch.tensor(unique_event_ids)
        else:
            self.sampling_order = torch.tensor(unique_event_ids[torch.randperm(len(unique_event_ids))])

    def _get_CSV_sample(self, sample_index,
                        position_offset = None,
                        **kwargs,
                        ):
        event_mask = self.data_table['event'] == sample_index
        event_rows = self.data_table[event_mask]
        
        segment_dtype = np.dtype([("x_start", "f4"),
                                  ("y_start", "f4"),
                                  ("z_start", "f4"),
                                  ("t_start", "f4"),
                                  ("x_end", "f4"),
                                  ("y_end", "f4"),
                                  ("z_end", "f4"),
                                  ("t_end", "f4"),
                                  ("dE", "f4"),
                                  ("dx", "f4"),
                                  ("dEdx", "f4")],
                                 align = True)

        start_4vec = torch.tensor((event_rows['startY']*mm,
                                   event_rows['startZ']*mm,
                                   event_rows['startX']*mm,
                                   event_rows['startT']*ns)).T
        end_4vec = torch.tensor((event_rows['endY']*mm,
                                 event_rows['endZ']*mm,
                                 event_rows['endX']*mm,
                                 event_rows['endT']*ns)).T
        
        dE = torch.tensor(event_rows['dE']*MeV)

        displacement = start_4vec[:,:3] - end_4vec[:,:3]
        dx = torch.sqrt(torch.sum(displacement**2, dim = 1))
        dEdx = torch.where(dx > 0, dE/dx, 0.)

        dQ = self.do_recombination(dE, dx, dEdx)
        charge_position, charge_time, charge_values = self.do_point_sampling(start_4vec,
                                                                             end_4vec,
                                                                             dx, dQ,
                                                                             )
        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
            
        return Track(charge_position + offset,
                     charge_time,
                     charge_values)

    def _get_CSV_meta(self, sample_index,
                      position_offset = None,
                      **kwargs):
        event_mask = self.data_table['event'] == sample_index
        event_rows = self.data_table[event_mask]

        kinetic_energy = -1
        charge = -1
        vertex = [-1, -1, -1]
        theta = -1
        phi = -1

        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
        
        meta_array = np.array([(sample_index,
                                kinetic_energy,
                                charge, # charge undefined
                                vertex[0] + offset[0],
                                vertex[1] + offset[1],
                                vertex[2] + offset[2],
                                theta, phi,
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
        return meta_array

    def get_sample(self, index, **kwargs):
        """
        parser.get_sample(index, **kwargs)

        Get the sample image from the loaded file.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        sample : Track object
            Return the loaded image as a point-sampled track object.
        
        """
        return self._get_CSV_sample(index, **kwargs)

    def get_meta(self, index, **kwargs):
        """
        parser.get_meta(index, **kwargs)

        Get the metadata from the loaded file

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        meta_array : array-like[float]
            Return an array containing the metadata for this sample.  The dtype
            of this array is defined in meta_dtype, above.
        
        """
        return self._get_CSV_meta(index, **kwargs)

class PenelopeParser (InputParser):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    def _open_file_handle(self, **kwargs):
        self.file_handle = np.load(self.input_filename, **kwargs)

    def _generate_sample_order(self, sequential_sampling):
        # These inputs have only one event per file
        # Sampling is trivial
        self.sampling_order = [0]
        
    def _get_penelope_sample(self, position_offset = None, **kwargs):
        charge_position = torch.tensor(self.file_handle['r']).T
        charge_values = torch.tensor(self.file_handle['num_e'])
        charge_time = torch.zeros_like(charge_values)
        
        offset = self.global_position_offset
        if position_offset:
            offset += torch.tensor(position_offset)
            
        return Track(charge_position + offset,
                     charge_time,
                     charge_values)

    def _get_penelope_meta(self,
                           **kwargs):
        # is the only available metadata for these energy in the filename?
        primary_energy = self.input_filename.split('/')[-1].split('_')[0][6:]
        try:
            primary_energy = float(primary_energy)*MeV
        except ValueError:
            primary_energy = -1
        
        meta_array = np.array([(0, # sample index (undefined)
                                primary_energy, # primary kinetic energy
                                -1, # charge (undefined)
                                0, 0, 0, # vertex position (undefined)
                                -1, -1, # primary attitude (undefined)
                                -1, # primary length (undefined)
                                )],
                              dtype = meta_dtype)
        return meta_array

    def get_sample(self, index, **kwargs):
        """
        parser.get_sample(index, **kwargs)

        Get the sample image from the loaded file.

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        sample : Track object
            Return the loaded image as a point-sampled track object.
        
        """
        return self._get_penelope_sample(**kwargs)
    
    def get_meta(self, index, **kwargs):
        """
        parser.get_meta(index, **kwargs)

        Get the metadata from the loaded file

        Parameters
        ----------
        index : int
            Index (in the file's internal scheme) of the sample to retrieve.

        Returns
        -------
        meta_array : array-like[float]
            Return an array containing the metadata for this sample.  The dtype
            of this array is defined in meta_dtype, above.
        
        """
        return self._get_penelope_meta()

parser_dict = {'root': RooTrackerParser,
               'edepsim': EdepSimParser,
               'marley': MarleyParser,
               'penelope': PenelopeParser,
               }
