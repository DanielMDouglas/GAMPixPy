from gampixpy import coordinates, analysis

import h5py
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")

def dtype_factory(readout_config):
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
                               ("exp x", "f4"),
                               ("exp y", "f4"),
                               ("exp z", "f4"),
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
                                ("exp x", "f4"),
                                ("exp y", "f4"),
                                ("exp z", "f4"),
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
                               ("exp x", "f4"),
                               ("exp y", "f4"),
                               ("exp z", "f4"),
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
                                ("exp x", "f4"),
                                ("exp y", "f4"),
                                ("exp z", "f4"),
                                ],
                               align = True)
        
    return tile_dtype, pixel_dtype

def main(args):

    op = analysis.OutputParser(args.input_gampixpy)

    conf = op.get_configs()
    
    infile = op._file_handle

    tile_dtype, pixel_dtype = dtype_factory(conf.readout_config)
    truth_tracking = conf.readout_config['truth_tracking']['enabled']

    coordinate_manager = coordinates.CoordinateManager(conf)
    
    pixel_coords = np.array([infile['pixels']['pixel x'],
                             infile['pixels']['pixel y'],
                             infile['pixels']['trig z'],
                             ]).T
    pixel_coords = torch.tensor(pixel_coords)
    pixel_exp_coords = coordinate_manager.to_experiment_coords(pixel_coords,
                                                               infile['pixels']['pixel tpc']
                                                               )

    coarse_coords = np.array([infile['tiles']['tile x'],
                              infile['tiles']['tile y'],
                              infile['tiles']['trig z'],
                              ]).T
    coarse_coords = torch.tensor(coarse_coords)
    coarse_exp_coords = coordinate_manager.to_experiment_coords(coarse_coords,
                                                                infile['tiles']['tile tpc']
                                                                )

    output_filename = args.output_file
    outfile = h5py.File(output_filename, 'w')

    outfile.copy(infile['meta'],
                 'meta')

    outfile.create_dataset('pixels',
                           shape = infile['pixels'].shape,
                           dtype = pixel_dtype,
                           maxshape = (None,))
    outfile.create_dataset('tiles',
                           shape = infile['tiles'].shape,
                           dtype = tile_dtype,
                           maxshape = (None,))

    outfile['pixels']['event id'] = infile['pixels']['event id']
    outfile['pixels']['pixel tpc'] = infile['pixels']['pixel tpc']
    outfile['pixels']['pixel x'] = infile['pixels']['pixel x']
    outfile['pixels']['pixel y'] = infile['pixels']['pixel y']
    outfile['pixels']['trig z'] = infile['pixels']['trig z']
    outfile['pixels']['trig t'] = infile['pixels']['trig t']
    outfile['pixels']['waveform'] = infile['pixels']['waveform']

    outfile['pixels']['exp x'] = pixel_exp_coords[:,0].cpu().numpy()
    outfile['pixels']['exp y'] = pixel_exp_coords[:,1].cpu().numpy()
    outfile['pixels']['exp z'] = pixel_exp_coords[:,2].cpu().numpy()

    outfile['tiles']['event id'] = infile['tiles']['event id']
    outfile['tiles']['tile tpc'] = infile['tiles']['tile tpc']
    outfile['tiles']['tile x'] = infile['tiles']['tile x']
    outfile['tiles']['tile y'] = infile['tiles']['tile y']
    outfile['tiles']['trig z'] = infile['tiles']['trig z']
    outfile['tiles']['trig t'] = infile['tiles']['trig t']
    outfile['tiles']['waveform'] = infile['tiles']['waveform']
    
    outfile['tiles']['exp x'] = coarse_exp_coords[:,0].cpu().numpy()
    outfile['tiles']['exp y'] = coarse_exp_coords[:,1].cpu().numpy()
    outfile['tiles']['exp z'] = coarse_exp_coords[:,2].cpu().numpy()

    if truth_tracking:
        outfile['pixels']['attribution'] = infile['pixels']['attribution']
        outfile['pixels']['label'] = infile['pixels']['label']
        outfile['tiles']['attribution'] = infile['tiles']['attribution']
        outfile['tiles']['label'] = infile['tiles']['label']

    outfile.close()
    
    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_gampixpy',
                        type = str,
                        help = 'input file to translate from gampixpy TPC coordinates to physical space')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        required = True,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')

    args = parser.parse_args()

    main(args)
