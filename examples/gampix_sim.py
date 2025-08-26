# This script is mean to demonstrate processing a single input
# from a provided input file, and showing how to use the built-in
# event display methods

from gampixpy import detector, input_parsing, plotting, config, output

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")

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

    # choose the correct input parser using the provided args
    # default value for `args.input_format` is 'edepsim'
    # so this will create an EdepSimParser object and expect hdf5 input
    input_parser = input_parsing.parser_dict[args.input_format](args.input_file,
                                                                readout_config = readout_config)

    event_data = input_parser.get_sample(args.event_index)
    # can also filter segments from the input to keep only specific pdg codes
    # (from an array or a singleton)
    # event_data = input_parser.get_sample(args.event_index, pdg_selection = 13)
    # event_data = input_parser.get_sample(args.event_index, pdg_selection = [11, 13, 22])

    event_meta = input_parser.get_meta(args.event_index)

    # call the detector sim in two steps:
    detector_model.drift(event_data) # generates drifted_track attribute
    detector_model.readout(event_data) # generates pixel_samples and coarse_tile_samples

    # # call the detector sim in one step:
    # detector_model.simulated(event_data)

    # inspect the simulation products
    # print (event_data.raw_track) # track after recombination and point sampling
    # print (event_data.drifted_track) # track after drifting (diffusion, attenuation)

    # make the event display
    evd = plotting.EventDisplay(event_data)

    evd.plot_raw_track() 
    # evd.plot_drifted_track()

    # methods where the z-axis is readout time
    # evd.plot_drifted_track_timeline()
    # evd.plot_drifted_track_timeline(alpha = 0) # can also pass kwargs to plt.scatter
    evd.plot_coarse_tile_measurement(readout_config,
                                     physics_config,
                                     detector_config,
                                     ) # plot tile hits
    evd.plot_pixel_measurement(readout_config,
                               physics_config,
                               detector_config,
                               ) # plot pixel hits
    # evd.plot_coarse_tile_measurement_timeline(readout_config) # plot tile hits
    # evd.plot_pixel_measurement_timeline(readout_config) # plot pixel hits

    evd.plot_drift_volumes(detector_config)
    
    evd.show()

    evd.save(args.plot_output)

    # save the simulation products to an hdf5 file
    if args.output_file:
        om = output.OutputManager(args.output_file)
        om.add_entry(event_data, event_meta)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        type = str,
                        help = 'input file from which to read and simulate an event')
    parser.add_argument('-i', '--input_format',
                        type = str,
                        default = 'edepsim',
                        help = 'input file format.  Must be one of {root, edepsim, marley}')
    parser.add_argument('-e', '--event_index',
                        type = int,
                        default = 5,
                        help = 'index of the event within the input file to be simulated')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')
    parser.add_argument('--plot_output',
                        type = str,
                        default = "",
                        help = 'file to save output plot')

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

    args = parser.parse_args()

    main(args)
