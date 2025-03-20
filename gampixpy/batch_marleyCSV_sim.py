import tqdm

import gampixpy
from gampixpy import detector, input_parsing, plotting, config, output

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

    if args.output_file:
        output_manager = output.OutputManager(args.output_file)

    input_parser = input_parsing.MarleyCSVParser(args.input_CSV_file)
    for event_index, marley_track, event_meta in tqdm.tqdm(input_parser):
        detector_model.simulate(marley_track)
        print ("found", len(marley_track.coarse_tiles_samples), "coarse tile hits")
        print ("found", len(marley_track.pixel_samples), "pixel hits")
        
        if args.output_file:
            output_manager.add_entry(marley_track, event_meta)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_CSV_file',
                        type = str,
                        help = 'input file from which to read and simulate an event')
    parser.add_argument('-e', '--event_index',
                        type = int,
                        default = 5,
                        help = 'index of the event within the input file to be simulated')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')

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
