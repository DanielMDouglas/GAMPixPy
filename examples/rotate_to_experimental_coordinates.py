import h5py
from gampixpy import config, coordinates, readout_objects
import numpy as np

import torch

N_LABELS_COARSE = readout_objects.N_LABELS_COARSE
N_LABELS_PIX = readout_objects.N_LABELS_PIX

pixel_dtype = np.dtype([("event id", "u4"),
                        ("hit x", "f4"),
                        ("hit y", "f4"),
                        ("hit z", "f4"),
                        ("tpc index", "u4"),
                        ("tpc x", "f4"),
                        ("tpc y", "f4"),
                        ("tpc z", "f4"),
                        ("hit t", "f4"),
                        ("hit charge", "f4"),
                        ("attribution", "f4", N_LABELS_COARSE),
                        ("label", "i4", N_LABELS_COARSE),
                        ],
                       align = True)
coarse_tile_dtype = np.dtype([("event id", "u4"),
                              ("hit x", "f4"),
                              ("hit y", "f4"),
                              ("hit z", "f4"),
                              ("tpc index", "u4"),
                              ("tpc x", "f4"),
                              ("tpc y", "f4"),
                              ("tpc z", "f4"),
                              ("hit t", "f4"),
                              ("hit charge", "f4"),
                              ("attribution", "f4", N_LABELS_PIX),
                              ("label", "i4", N_LABELS_PIX),
                              ],
                             align = True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")

def main(args):

    detector_config = config.DetectorConfig(args.detector_config)
    coordinate_manager = coordinates.CoordinateManager(detector_config)

    infile = h5py.File(args.input_gampixpy)
    
    pixel_coords = np.array([infile['pixel_hits']['pixel x'],
                             infile['pixel_hits']['pixel y'],
                             infile['pixel_hits']['hit z'],
                             ]).T
    pixel_coords = torch.tensor(pixel_coords)
    pixel_exp_coords = coordinate_manager.to_experiment_coords(pixel_coords,
                                                               infile['pixel_hits']['pixel tpc']
                                                               )

    coarse_coords = np.array([infile['coarse_hits']['tile x'],
                              infile['coarse_hits']['tile y'],
                              infile['coarse_hits']['hit z'],
                              ]).T
    coarse_coords = torch.tensor(coarse_coords)
    coarse_exp_coords = coordinate_manager.to_experiment_coords(coarse_coords,
                                                                infile['coarse_hits']['tile tpc']
                                                                )

    output_filename = args.output_file
    outfile = h5py.File(output_filename, 'w')

    outfile.copy(infile['meta'],
                 'meta')

    outfile.create_dataset('pixel_hits',
                           shape = infile['pixel_hits'].shape,
                           dtype = pixel_dtype,
                           maxshape = (None,))
    outfile.create_dataset('coarse_hits',
                           shape = infile['coarse_hits'].shape,
                           dtype = coarse_tile_dtype,
                           maxshape = (None,))

    outfile['pixel_hits']['event id'] = infile['pixel_hits']['event id']
    outfile['pixel_hits']['hit x'] = pixel_exp_coords[:,0].cpu().numpy()
    outfile['pixel_hits']['hit y'] = pixel_exp_coords[:,1].cpu().numpy()
    outfile['pixel_hits']['hit z'] = pixel_exp_coords[:,2].cpu().numpy()
    outfile['pixel_hits']['tpc index'] = infile['pixel_hits']['pixel tpc']
    outfile['pixel_hits']['tpc x'] = pixel_coords[:,0].cpu().numpy()
    outfile['pixel_hits']['tpc y'] = pixel_coords[:,1].cpu().numpy()
    outfile['pixel_hits']['tpc z'] = pixel_coords[:,2].cpu().numpy()
    outfile['pixel_hits']['hit t'] = infile['pixel_hits']['hit t']
    outfile['pixel_hits']['hit charge'] = infile['pixel_hits']['hit charge']
    outfile['pixel_hits']['attribution'] = infile['pixel_hits']['attribution']
    outfile['pixel_hits']['label'] = infile['pixel_hits']['label']

    outfile['coarse_hits']['event id'] = infile['coarse_hits']['event id']
    outfile['coarse_hits']['hit x'] = coarse_exp_coords[:,0].cpu().numpy()
    outfile['coarse_hits']['hit y'] = coarse_exp_coords[:,1].cpu().numpy()
    outfile['coarse_hits']['hit z'] = coarse_exp_coords[:,2].cpu().numpy()
    outfile['coarse_hits']['tpc index'] = infile['coarse_hits']['tile tpc']
    outfile['coarse_hits']['tpc x'] = coarse_coords[:,0].cpu().numpy()
    outfile['coarse_hits']['tpc y'] = coarse_coords[:,1].cpu().numpy()
    outfile['coarse_hits']['tpc z'] = coarse_coords[:,2].cpu().numpy()
    outfile['coarse_hits']['hit t'] = infile['coarse_hits']['hit t']
    outfile['coarse_hits']['hit charge'] = infile['coarse_hits']['hit charge']
    outfile['coarse_hits']['attribution'] = infile['coarse_hits']['attribution']
    outfile['coarse_hits']['label'] = infile['coarse_hits']['label']

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

    parser.add_argument('-d', '--detector_config',
                        type = str,
                        required = True,
                        help = 'detector configuration yaml')

    args = parser.parse_args()

    main(args)
