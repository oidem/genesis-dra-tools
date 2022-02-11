# program msm_analysis.py

import sys, math, os
import matplotlib.pyplot as plt
import numpy as np
import pyemma
from pyemma.util.contexts import settings
import argparse
import time
import copy

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input feature data')
    parser.add_argument('--dtraj', dest = 'dtraj_file', required = True, help = 'dtraj file of input feature data')
    parser.add_argument('--o', dest = 'out_name', default = 'its_result', help = 'root name for output files')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--lag', dest = 'lag_time', default = '0', help = 'lag time step for VAMP2 score', type = int)
    parser.add_argument('--mode', dest = 'n_mode', default = '5', help = 'number of modes in MSM you want to plot', type = int)
    parser.add_argument('--trial', dest = 'n_trial', default = '5', help = 'number of trials in calculation of vamp', type = int)
    parser.add_argument('--xmin', dest = 'x_min', default = '0', help = 'minimum value of x axis', type = float)
    parser.add_argument('--xmax', dest = 'x_max', default = '0', help = 'maximum value of x axis', type = float)
    parser.add_argument('--xtic', dest = 'x_del', default = '0', help = 'step value of x axis', type = float)
    parser.add_argument('--ymin', dest = 'y_min', default = '0', help = 'minimum value of y axis', type = float)
    parser.add_argument('--ymax', dest = 'y_max', default = '0', help = 'maximum value of y axis', type = float)
    parser.add_argument('--ytic', dest = 'y_del', default = '0', help = 'step value of y axis', type = float)
    parser.add_argument('--score', dest = 'score_method', default = '', help = 'scoring method for msm consctructed by discretized trajectory: VAMP1 or VAMP2')

    # optional variables for transforming new data
    parser.add_argument('--figsize', dest = 'fig_size', default = '5', help = 'size of x-dimension in inch (size of y-dimension will be determined by aspect ratio)', type = float)
    parser.add_argument('--dpi', dest = 'dpi', default = '300', help = 'dpi', type = int)
    parser.add_argument('--square', dest = 'do_square', help='ignore the aspect ratio of xrange/yange', action='store_true')

    args = parser.parse_args()

    fn_in        = args.in_file
    fn_dtraj     = args.dtraj_file
    fn_out       = args.out_name
    is_multiple  = args.is_multiple
    skip_output  = args.skip_output
    verbose      = args.verbose
    lag          = args.lag_time
    nmode        = args.n_mode
    ntrial       = args.n_trial
    xmin         = args.x_min
    xmax         = args.x_max
    xdel         = args.x_del
    ymin         = args.y_min
    ymax         = args.y_max
    ydel         = args.y_del
    score_method = args.score_method

    fig_size     = args.fig_size
    dpi          = args.dpi
    square       = args.do_square

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    dtraj file: {}'.format(fn_dtraj))
    print('    output prefix: {}'.format(fn_out))
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('its options')
    print('    lag time: {}'.format(lag))
    print('    number of modes: {}'.format(nmode))
    print('    number of trials of msm calculation: {}'.format(ntrial))
    print('    minimum value of x-axis: {}'.format(xmin))
    print('    maximum value of x-axis: {}'.format(xmax))
    print('    step value of x-axis: {}'.format(xdel))
    print('    minimum value of y-axis: {}'.format(ymin))
    print('    maximum value of y-axis: {}'.format(ymax))
    print('    step value of y-axis: {}'.format(ydel))

    print('expert options')
    print('    figure size (inch): {}'.format(fig_size))
    print('    dpi: {}'.format(dpi))
    if (square):
        print('    ignore the aspect ratio of xrange/yrange?: yes')
    else:
        print('    ignore the aspect ratio of xrange/yrange?: no')

    start = time.time()

    if (score_method == ''):
        score_method = 'VAMP2'
    elif (score_method == 'vamp1'):
        score_method = 'VAMP1'
    elif (score_method == 'vamp2'):
        score_method = 'VAMP2'
    elif ((score_method == 'VAMP1') or (score_method == 'VAMP2')):
        pass
    else:
        print('Error: invalid argument of --score option.')
        print('        you can specify only "VAMP1" or "VAMP2" ("vamp1" and "vamp2" are also OK!).')
        sys.exit()

    # read input feature data
    infiles = []
    if (is_multiple):
        with open(fn_in) as input:
            lines = input.readlines()
            for line in lines:
                infiles.append(line.replace('\n', ''))  # stripping line ending
    else:
        infiles.append(fn_in)

    nfile = len(infiles)

    if (verbose):
        print('')
        print('number of input files: {}'.format(nfile))

    data = []

    for infile in infiles:
        temparray = np.loadtxt(infile)
        data.append(np.delete(temparray, [0], axis=1))

    print('type of data: ', type(data))
    print('dimensions: ', len(data))
    print('type of data[0]', type(data[0]))
    print('shape of elements: ', data[0].shape)

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)

    data_concatenated = np.concatenate(data)

    # read dtraj of input data
    dtrajfiles = []
    if (is_multiple):
        with open(fn_dtraj) as input:
            lines = input.readlines()
            for line in lines:
                dtrajfiles.append(line.replace('\n', ''))  # stripping line ending
    else:
        dtrajfiles.append(fn_dtraj)

    ndfile = len(dtrajfiles)

    if (verbose):
        print('')
        print('number of dtraj files: {}'.format(ndfile))

    dtraj = []

    for dtrajfile in dtrajfiles:

        temparray = np.loadtxt(dtrajfile, dtype='int')
        temparray2 = np.delete(temparray, [0], axis=1)
        dtraj.append(temparray2.squeeze())

    print('type of dtraj data: ', type(dtraj))
    print('dimensions: ', len(dtraj))
    print('type of dtraj data[0]', type(dtraj[0]))
    print('shape of elements: ', dtraj[0].shape)

    dtraj_concatenated = np.concatenate(dtraj)

    if (verbose):
        print('')
        print('shape of each dtraj data:')
        for i in range(len(dtraj)):
            print('    ', i, ' ', dtraj[i].shape)

    scores = np.zeros(ntrial)
    score_max = 0.0

    if (verbose):
        print('')
        print('trial number, VAMP1 ( = GMRQ) score')

    for i in range(ntrial):
        with pyemma.util.contexts.settings(show_progress_bars=False):
            msm = pyemma.msm.estimate_markov_model(dtraj, lag=lag)
            scores[i] = msm.score_cv(dtraj, n=1, score_method=score_method)
            if (verbose):
                print(i, '   ', scores[i])

            if (scores[i] > score_max):
                msm_best = copy.deepcopy(msm)
                score_max = scores[i]

    eigval = msm_best.eigenvalues()
    eigvec_r = msm_best.eigenvectors_right()
    eigvec_l = msm_best.eigenvectors_left()

    print('')
    print('fraction of states used = {:f}'.format(msm_best.active_state_fraction))
    print('fraction of counts used = {:f}'.format(msm_best.active_count_fraction))

    print('')
    print('first eigenvector is one: {} (min={}, max={})'.format(np.allclose(eigvec_r[:, 0], 1, atol=1e-15), eigvec_r[:, 0].min(), eigvec_r[:, 0].max()))

    if (not skip_output):

        plt.rcParams['figure.dpi'] = dpi

        xsize = fig_size
        ysize = ((ymax - ymin) / (xmax - xmin)) * xsize
        if (square):
            ysize = xsize

        if (xdel > 0 ):
            interval_x = np.arange(xmin, xmax, xdel)
            plt.xticks(interval_x)
        if (ydel > 0):
            interval_y = np.arange(ymin, ymax, ydel)
            plt.yticks(interval_y)

        fig = plt.figure(figsize=(xsize, ysize))

        if (xmin != xmax) and (xmin < xmax):
            plt.xlim([xmin, xmax])
        if (ymin != ymax) and (ymin < ymax):
            plt.ylim([ymin, ymax])        

        fn_png = fn_out + '_stationary_plot.png'
        misc = {}
        # plot stationary distribution
        fig, ax, misc = pyemma.plots.plot_contour(*data_concatenated[:, :2].T, msm_best.pi[dtraj_concatenated], cbar_label='stationary_distribution', method='nearest', mask=True)
        #ax.scatter(*cluster.clustercenters.T, s=15, c='C1')
        ax.set_xlabel('Component1')
        ax.set_ylabel('Component2')
        fig.tight_layout()
        fig.savefig(fn_png)
        fig.delaxes(fig.axes[1])

        # plot eigenvectors
        for i in range(nmode):
            idx = i + 2
            fn_png = fn_out + '_eigenvector' + format(idx, '03d') + '_plot.png'
            misc = {}
            fig, ax, misc = pyemma.plots.plot_contour(*data_concatenated[:, :2].T, eigvec_r[dtraj_concatenated, i + 1], cmap='PiYG', cbar_label='{}. right eigenvector'.format(i + 2), mask=True)
            #ax.scatter(*cluster.clustercenters.T, s=15, c='C1')
            ax.set_xlabel('Component1')
            ax.set_ylabel('Component2')
            fig.tight_layout()
            fig.savefig(fn_png)
            fig.delaxes(fig.axes[1])

        # output centers of clusters
#        f = open(fn_center, "w")
#        for i in range(nclass):
#            idx = i + 1
#            line = format(idx, '>9')
#            for j in range(ncomp):
#                if (j == 0):
#                    line += format(center_out[i,j], '>10.5f')
#                else:
#                    line += format(center_out[i,j], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
