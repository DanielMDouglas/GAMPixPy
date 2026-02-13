from gampixpy import detector, generator, input_parsing, plotting, config, output, coordinates
from gampixpy.units import *

import numpy as np
import tqdm
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")

def first_guess_heuristics(sim_readout, physics_config, detector_config):

    coordinate_manager = coordinates.CoordinateManager(detector_config)
    
    t_guess = 0*us

    coarse_hits_q = np.array([hit.coarse_cell_measurement
                             for hit in sim_readout.coarse_tiles_samples])
    
    pixel_hits_tpc = np.array([hit.pixel_tpc
                               for hit in sim_readout.pixel_samples])
    pixel_hits_q = np.array([hit.hit_measurement
                             for hit in sim_readout.pixel_samples])
    pixel_hits_x = np.array([hit.pixel_pos[0]
                             for hit in sim_readout.pixel_samples])
    pixel_hits_y = np.array([hit.pixel_pos[1]
                             for hit in sim_readout.pixel_samples])
    pixel_hits_t = np.array([hit.hit_timestamp
                             for hit in sim_readout.pixel_samples])

    q_guess = int(np.sum(coarse_hits_q))
    
    x_guess_tpc = np.sum(pixel_hits_x*pixel_hits_q)/np.sum(pixel_hits_q)
    y_guess_tpc = np.sum(pixel_hits_y*pixel_hits_q)/np.sum(pixel_hits_q)

    weights = pixel_hits_q/np.sum(pixel_hits_q)
    mean_arrival_time = np.sum(pixel_hits_t*weights)
    std_arrival_time = np.sqrt(np.sum((pixel_hits_t - mean_arrival_time)**2*weights))
    v = physics_config['charge_drift']['drift_speed']
    DL = physics_config['charge_drift']['diffusion_longitudinal']
    depth_guess = v**3*std_arrival_time**2/(2*DL)
    
    guess_coords_tpc = torch.tensor([x_guess_tpc,
                                     y_guess_tpc,
                                     depth_guess])
    
    guess_coords_exp = coordinate_manager.to_experiment_coords(guess_coords_tpc, 0)
    guess_coords_exp = guess_coords_exp.cpu().numpy()

    x_guess = guess_coords_exp[0]
    y_guess = guess_coords_exp[1]
    z_guess = guess_coords_exp[2]
    
    return x_guess, y_guess, z_guess, t_guess, q_guess

def main(args):
    # load configs for physics, detector, and readout

    if args.detector_config == "":
        detector_config = config.default_detector_params
    else:
        detector_config = config.DetectorConfig(args.detector_config)

    if args.physics_config == "":
        physics_config = config.default_physics_params
    else:
        physics_config = config.PhysicsConfig(args.physics_config)

    if args.readout_config == "":
        readout_config = config.default_readout_params
    else:
        readout_config = config.ReadoutConfig(args.readout_config)

    detector_model = detector.DetectorModel(detector_params = detector_config,
                                            physics_params = physics_config,
                                            readout_params = readout_config,
                                            )

    x_range = args.x_range.split(',')
    x_range = [float(x_range[0]), float(x_range[1])]

    y_range = args.y_range.split(',')
    y_range = [float(y_range[0]), float(y_range[1])]

    z_range = args.z_range.split(',')
    z_range = [float(z_range[0]), float(z_range[1])]

    t_range = args.t_range.split(',')
    t_range = [float(t_range[0]), float(t_range[1])]

    q_range = args.q_range.split(',')
    q_range = [float(q_range[0]), float(q_range[1])]

    ps_generator = generator.PointSource(x_range = x_range,
                                         y_range = y_range,
                                         z_range = z_range,
                                         t_range = t_range,
                                         q_range = q_range,
                                         )

    out_array = np.empty((args.n_samples, 10))

    for i in tqdm.tqdm(range(args.n_samples)):
        sim_track = ps_generator.get_sample()

        detector_model.simulate(sim_track,
                                verbose = False)

        print ("sim_track truth: ",
               ps_generator.x_init,
               ps_generator.y_init,
               ps_generator.z_init,
               ps_generator.t_init,
               ps_generator.q_init,
               )

        init_guess = first_guess_heuristics(sim_track,
                                            physics_config,
                                            detector_config,
                                            )
        print (init_guess)

        out_array[i, 0] = ps_generator.x_init
        out_array[i, 1] = ps_generator.y_init
        out_array[i, 2] = ps_generator.z_init
        out_array[i, 3] = ps_generator.t_init
        out_array[i, 4] = ps_generator.q_init
        out_array[i, 5] = init_guess[0]
        out_array[i, 6] = init_guess[1]
        out_array[i, 7] = init_guess[2]
        out_array[i, 8] = init_guess[3]
        out_array[i, 9] = init_guess[4]

        # if args.output_file:
        #     om.add_entry(cloud_track, cloud_meta)

    np.save(args.output_file, out_array)
    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')
    parser.add_argument('-n', '--n_samples',
                        type = int,
                        default = 1000,
                        help = 'number of point sources per output file')

    parser.add_argument('-d', '--detector_config',
                        type = str,
                        default = "",
                        help = 'detector configuration yaml')
    parser.add_argument('-p', '--physics_config',
                        type = str,
                        default = "",
                        help = 'physics configuration yaml')
    parser.add_argument('-r', '--readout_config',
                        type = str,
                        default = "",
                        help = 'readout configuration yaml')

    parser.add_argument('-x', '--x_range',
                        type = str,
                        default = "-10,10",
                        help = 'min,max x values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-y', '--y_range',
                        type = str,
                        default = "-10,10",
                        help = 'min,max y values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-z', '--z_range',
                        type = str,
                        default = "10,100",
                        help = 'min,max z values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-t', '--t_range',
                        type = str,
                        default = "0,0",
                        help = 'min,max t values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-q', '--q_range',
                        type = str,
                        default = "100,100000",
                        help = 'min,max q values over which to generate point sources (e.g. -2,4)')

    args = parser.parse_args()

    main(args)
