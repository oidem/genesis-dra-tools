# program tica_interface.py

import sys, math, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyemma
from pyemma.util.contexts import settings
import argparse
import time
import copy

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input dtraj data')
    parser.add_argument('--o', dest = 'out_name', default = 'its_result', help = 'root name for output files')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')    
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--lag', dest = 'lag_time', required = True, help = 'lag time step for MSM construction', type = int)
    parser.add_argument('--state', dest = 'n_state', required = True, help = 'number of metastable states', type = int)
    parser.add_argument('--level', dest = 'conf_level', default = '0.95', help = 'confidence level of predicted MSM', type = float)
    parser.add_argument('--estimate', dest = 'only_estimate', help = 'just only estimation of number of metastable states', action = 'store_true')
    parser.add_argument('--data', dest = 'data_file', default = None, help = 'data list (or file) used in microstate partitioning')
    parser.add_argument('--center', dest = 'center_file', default = None, help = 'center coordinate file of microstate partitioning')
    

    args = parser.parse_args()

    fn_in         = args.in_file
    fn_out        = args.out_name
    is_multiple   = args.is_multiple
    skip_output   = args.skip_output
    verbose       = args.verbose
    lag           = args.lag_time
    nstate        = args.n_state
    conf          = args.conf_level
    only_estimate = args.only_estimate
    fn_data       = args.data_file
    fn_center     = args.center_file
    
    if (only_estimate) and (fn_data is None or fn_center is None):
        print('Error: If you want to estimate the number of metastable states, you must specify "--data" and "--center" options.')
        sys.exit()

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output prefix: {}'.format(fn_out))
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('CK-test options')
    print('    lag time: {}'.format(lag))
    print('    number of states: {}'.format(nstate))
    if (only_estimate):
        print('    only estimate the number of metastable states?: yes')
        print('    data file used in microstate partitioning: {}'.format(fn_data))
        print('    center coordinate file of microstate partitioning: {}'.format(fn_center))
    else:
        print('    only estimate the number of metastable states?: no')
        print('    confidence level of predicted MSM: {}'.format(conf))

    start = time.time()

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

        temparray = np.loadtxt(infile, dtype='int')
        temparray2 = np.delete(temparray, [0], axis=1)
        data.append(temparray2.squeeze())

    print('type of data: ', type(data))
    print('dimensions: ', len(data))
    print('type of data[0]', type(data[0]))
    print('shape of elements: ', data[0].shape)

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)

    rawdata = []
    dtraj_center = None
    rawdata_concatenated = None
    dtraj_concatenated = None
    if (only_estimate):

        rawdatafiles = []
        if (is_multiple):
            with open(fn_data) as input:
                lines = input.readlines()
                for line in lines:
                    rawdatafiles.append(line.replace('\n', ''))  # stripping line ending
        else:
            rawdatafilesfiles.append(fn_data)
        
        for datafile in rawdatafiles:
            temparray = np.loadtxt(datafile)
            rawdata.append(np.delete(temparray, [0], axis=1))

        temparray = np.loadtxt(fn_center)
        dtraj_center = np.delete(temparray, [0], axis = 1)

        dtraj_concatenated = np.concatenate(data)
        rawdata_concatenated = np.concatenate(rawdata)


    if (only_estimate):

        msm = pyemma.msm.estimate_markov_model(data, lag=lag, dt_traj='1 step')

        print('fraction of states used = {:f}'.format(msm.active_state_fraction))
        print('fraction of counts used = {:f}'.format(msm.active_count_fraction))

        fn_stat = fn_out + '_stationary_distribution.png'

        print('test')
        print(type(msm.pi))
        print(msm.pi.shape)

        fig, ax = plt.subplots(figsize = (5, 5))
        pyemma.plots.plot_contour(*rawdata_concatenated.T, msm.pi[dtraj_concatenated], ax=ax, cbar_label='stationary_distribution', method='nearest', mask=True)
        ax.scatter(*dtraj_center.T, s=15, c='C1')
        ax.set_xlabel('1st component')
        ax.set_ylabel('2nd component')
        fig.tight_layout()
        fig.savefig(fn_stat)

        eigvec = msm.eigenvectors_right()
        print('first eigenvector is one: {} (min={}, max={})'.format(np.allclose(eigvec[:, 0], 1, atol=1e-15), eigvec[:, 0].min(), eigvec[:, 0].max()))

        fn_eigvec = fn_out + '_right_eigvec.png'

        row = 1
        col = 1
        xsize = 1
        ysize = 1
        if (nstate < 5):
            row = 1
            col = nstate
            xsize = 4 * col
            ysize = 3 * row
        elif ((nstate % 3) == 0):
            row = nstate / 3
            col = 3
            xsize = 4 * col
            ysize = 3 * row
        else:
            row = (nstate // 4) + 1
            col = 4
            xsize = 4 * col
            ysize = 3 * row

        fig, axes = plt.subplots(int(row), int(col), figsize=(xsize, ysize))
        for i, ax in enumerate(axes.flat):
            fig, ax = pyemma.plots.plot_contour(*rawdata_concatenated.T, eigvec[dtraj_concatenated, i + 1], cmap='PiYG', cbar_label='{}. right eigenvector'.format(i + 2), mask=True)
            ax.scatter(*cluster.clustercenters.T, s=15, c='C1')
            ax.set_xlabel('1st component')
            ax.set_ylabel('2nd component')
        fig.tight_layout()
        fig.savefig(fn_eigvec)

        sys.exit()

    msm = pyemma.msm.bayesian_markov_model(data, lag=lag, dt_traj='1 step', conf=conf)
    cktest = msm.cktest(nstate)

    if (not skip_output):

        fn_png = fn_out + '_cktest_plot.png'

        # plot
        fig = pyemma.plots.plot_cktest(cktest, units='steps', dt=1)
        #fig[0].tight_layout()
        fig[0].savefig(fn_png)

#        fn_mean = fn_out + '_its_sample_mean.dat'
#        fn_std = fn_out + '_its_sample_std.dat'
#        fn_its = fn_out + '_its.dat'
#        fn_conf_lower = fn_out + '_its_conf_lower.dat'
#        fn_conf_upper = fn_out + '_its_conf_upper.dat'
#
#        f = open(fn_mean, "w")
#        for i in range(len(lags)):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(lags[i], '>10')
#            for j in range(nits):
#                line += format(its.sample_mean[i,j], '>11.5f')
#
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        f = open(fn_std, "w")
#        for i in range(len(lags)):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(lags[i], '>10')
#            for j in range(nits):
#                line += format(its.sample_std[i,j], '>11.5f')
#
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        f = open(fn_its, "w")
#        for i in range(len(lags)):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(lags[i], '>10')
#            for j in range(nits):
#                line += format(its.timescales[i,j], '>11.5f')
#
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        its_conf = its.get_sample_conf()
#
#        temparray = its_conf[0]
#        f = open(fn_conf_lower, "w")
#        for i in range(len(lags)):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(lags[i], '>10')
#            for j in range(nits):
#                line += format(temparray[i,j], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        temparray = its_conf[1]
#        f = open(fn_conf_upper, "w")
#        for i in range(len(lags)):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(lags[i], '>10')
#            for j in range(nits):
#                line += format(temparray[i,j], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
