# This script is meant to demonstrate batch processing of
# an entire input file, and producing a single output file 
# with simultion products and metadata with corresponding
# event indices 

import tqdm

import gampixpy
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

    if args.output_file:
        output_manager = output.OutputManager(args.output_file)

    input_parser = input_parsing.parser_dict[args.input_format](args.input_edepsim_file,
                                                                readout_config = readout_config)

    # # When you need only specific segments from an input event,
    # # you can specify a whitelist of PDG codes to pass through 
    # for event_index in tqdm.tqdm(input_parser.sampling_order):
    #     edepsim_track = input_parser.get_sample(event_index.item(),
    #                                             pdg_selection = [-13, 11])
    #     event_meta = input_parser.get_meta(event_index.item())

    # If you just need all segments, use the iter method instead
    for event_index, edepsim_track, event_meta in tqdm.tqdm(input_parser):

        detector_model.simulate(edepsim_track,
                                verbose = False)

        if args.output_file:
            output_manager.add_entry(edepsim_track,
                                     event_meta,
                                     # event_id = event_index.cpu().item())
                                     event_id = event_index)

        if args.n_samples != -1:
            if event_index >= args.n_samples:
                print ("done at", event_index)
                break

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_edepsim_file',
                        type = str,
                        help = 'input file from which to read and simulate an event')
    parser.add_argument('-i', '--input_format',
                        type = str,
                        default = 'edepsim',
                        help = 'input file format.  Must be one of {root, edepsim, marley}')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')
    parser.add_argument('-n', '--n_samples',
                        type = int,
                        default = -1,
                        help = 'Number of samples to process (default is to process all events)')

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
