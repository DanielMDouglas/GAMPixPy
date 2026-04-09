import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st

depth_bin_edges = np.linspace(0, 650, 26)
depth_bin_centers = 0.5*(depth_bin_edges[1:] + depth_bin_edges[:-1])

def plot_transverse_error(ax, sim_file_name, label):
    d = np.load(sim_file_name)

    true_x = d[:,0]
    true_y = d[:,1]
    true_z = d[:,2]
    true_t = d[:,3]
    true_q = d[:,4]
        
    reco_x = d[:,5]
    reco_y = d[:,6]
    reco_z = d[:,7]
    reco_t = d[:,8]
    reco_q = d[:,9]

    depths = 325 - true_x
    true_transverse_pos = np.stack((true_y, true_z))
    reco_transverse_pos = np.stack((reco_y, reco_z))
    
    dist = np.sqrt(np.sum(np.power(true_transverse_pos - reco_transverse_pos, 2), axis = 0))

    bs = st.binned_statistic(depths,
                             dist,
                             statistic = 'mean',
                             bins = depth_bin_edges)
    # ax.scatter(depths,
    #            dist,
    #            label = label)
    ax.plot(depth_bin_centers,
            bs[0],
            label = label)

def plot_depth_estimate(ax, sim_file_name, label):
    d = np.load(sim_file_name)

    true_x = d[:,0]
    true_y = d[:,1]
    true_z = d[:,2]
    true_t = d[:,3]
    true_q = d[:,4]
        
    reco_x = d[:,5]
    reco_y = d[:,6]
    reco_z = d[:,7]
    reco_t = d[:,8]
    reco_q = d[:,9]

    true_depth = 325 - true_x
    reco_depth = 325 - reco_x
    
    bs_mean = st.binned_statistic(true_depth,
                                  reco_depth,
                                  statistic = 'mean',
                                  bins = depth_bin_edges)
    bs_upper_q = st.binned_statistic(true_depth,
                                     reco_depth,
                                     statistic = lambda x: st.quantile(x, 0.84),
                                     bins = depth_bin_edges)
    bs_lower_q = st.binned_statistic(true_depth,
                                     reco_depth,
                                     statistic = lambda x: st.quantile(x, 0.16),
                                     bins = depth_bin_edges)
    # ax.scatter(true_depth,
    #            reco_depth,
    #            label = label)
    ax.fill_between(depth_bin_centers,
                    bs_lower_q[0],
                    bs_upper_q[0],
                    alpha = 0.5)
    ax.plot(depth_bin_centers,
            bs_mean[0],
            label = label)

def plot_depth_width(ax, sim_file_name, label):
    d = np.load(sim_file_name)

    true_x = d[:,0]
    true_y = d[:,1]
    true_z = d[:,2]
    true_t = d[:,3]
    true_q = d[:,4]
        
    reco_x = d[:,5]
    reco_y = d[:,6]
    reco_z = d[:,7]
    reco_t = d[:,8]
    reco_q = d[:,9]

    true_depth = 325 - true_x
    reco_depth = 325 - reco_x
    
    bs_std = st.binned_statistic(true_depth,
                                 reco_depth,
                                 statistic = 'std',
                                 bins = depth_bin_edges)
    # ax.scatter(true_depth,
    #            reco_depth,
    #            label = label)
    ax.plot(depth_bin_centers,
            bs_std[0],
            label = label)

def main(args):

    transverse_fig = plt.figure()
    transverse_ax = transverse_fig.gca()

    depth_fig = plt.figure()
    depth_ax = depth_fig.gca()

    depth_width_fig = plt.figure()
    depth_width_ax = depth_width_fig.gca()

    file_names = ['ps_1e7.npy',
                  'ps_5e6.npy',
                  'ps_2e6.npy',
                  'ps_1e6.npy',
                  'ps_5e5.npy',
                  'ps_2e5.npy',
                  'ps_1e5.npy',
                  'ps_5e4.npy',
                  'ps_2e4.npy',
                  'ps_1e4.npy',
                  ]
    labels = [r'$Q = 1 \times 10^7$e',
              r'$Q = 5 \times 10^6$e',
              r'$Q = 2 \times 10^6$e',
              r'$Q = 1 \times 10^6$e',
              r'$Q = 5 \times 10^5$e',
              r'$Q = 2 \times 10^5$e',
              r'$Q = 1 \times 10^5$e',
              r'$Q = 5 \times 10^4$e',
              r'$Q = 2 \times 10^4$e',
              r'$Q = 1 \times 10^4$e',
              ]

    for file_name, label in zip(file_names, labels):
        plot_transverse_error(transverse_ax, file_name, label)
        plot_depth_estimate(depth_ax, file_name, label)
        plot_depth_width(depth_width_ax, file_name, label)
        
    transverse_ax.semilogy()
    transverse_ax.set_xlabel(r'Point Source Depth [cm]')
    transverse_ax.set_ylabel(r'Estimated Transverse Position Error [cm]')

    depth_ax.set_xlabel(r'True Point Source Depth [cm]')
    depth_ax.set_ylabel(r'Estimated Depth [cm]')
    depth_ax.plot([0, 650],
                  [0, 650],
                  ls = '--',
                  color = 'red')

    depth_width_ax.semilogy()
    depth_width_ax.set_xlabel(r'True Point Source Depth [cm]')
    depth_width_ax.set_ylabel(r'Estimated Depth Resolution [cm]')

    transverse_ax.legend()
    depth_ax.legend()
    depth_width_ax.legend()
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')

    args = parser.parse_args()

    main(args)

