# program tica_interface.py

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
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input dtraj data')
    parser.add_argument('--o', dest = 'out_name', default = 'its_result', help = 'root name for output files')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--lag', dest = 'lag_time', default = '0', help = 'lag time step for VAMP2 score', type = int)
    parser.add_argument('--its', dest = 'n_its', default = '10', help = 'number of classes (default: 10)', type = int)
    parser.add_argument('--bayes', dest = 'bayes', help = 'use Bayesian to calculate uncertainity of its', action = 'store_true')

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    verbose     = args.verbose
    lag         = args.lag_time
    nits        = args.n_its
    bayes       = args.bayes

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output prefix: {}'.format(fn_out))
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('its options')
    print('    lag time: {}'.format(lag))
    print('    number of states: {}'.format(nits))
    if (bayes):
        print('    use Bayesian for uncertainity calculation?: yes')
    else:
        print('    use Bayesian for uncertainity calculation?: no')

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

    lags = []
    if (lag == 0):
        lags = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    else:
        lag_temp = 0
        i = 0
        while (lag_temp < lag):
            lag_temp = float(2 ** i)
            lags.append(int(lag_temp))
            i += 1
        lag_temp = float(2 ** i)
        lags.append(int(lag_temp))

    print('')
    print('list of lag time:')
    print(lags)

    error = None
    if (bayes):
        error = 'bayes'

    its = pyemma.msm.its(data, lags=lags, nits=nits, errors=error)

    if (not skip_output):

        fn_png = fn_out + '_its_plot.png'

        # plot
        fig, ax = plt.subplots(figsize = (8, 6))
        pyemma.plots.plot_implied_timescales(its, ax=ax, units='steps', dt=1)
        fig.tight_layout()
        fig.savefig(fn_png)

        fn_mean = fn_out + '_its_sample_mean.dat'
        fn_std = fn_out + '_its_sample_std.dat'
        fn_its = fn_out + '_its.dat'
        fn_conf_lower = fn_out + '_its_conf_lower.dat'
        fn_conf_upper = fn_out + '_its_conf_upper.dat'

        f = open(fn_mean, "w")
        for i in range(len(lags)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(lags[i], '>10')
            for j in range(nits):
                line += format(its.sample_mean[i,j], '>11.5f')

            f.write(line)
            f.write("\n")
        f.close()

        f = open(fn_std, "w")
        for i in range(len(lags)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(lags[i], '>10')
            for j in range(nits):
                line += format(its.sample_std[i,j], '>11.5f')

            f.write(line)
            f.write("\n")
        f.close()

        f = open(fn_its, "w")
        for i in range(len(lags)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(lags[i], '>10')
            for j in range(nits):
                line += format(its.timescales[i,j], '>11.5f')

            f.write(line)
            f.write("\n")
        f.close()

        its_conf = its.get_sample_conf()

        temparray = its_conf[0]
        f = open(fn_conf_lower, "w")
        for i in range(len(lags)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(lags[i], '>10')
            for j in range(nits):
                line += format(temparray[i,j], '>11.5f')
        
            f.write(line)
            f.write("\n")
        f.close()

        temparray = its_conf[1]
        f = open(fn_conf_upper, "w")
        for i in range(len(lags)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(lags[i], '>10')
            for j in range(nits):
                line += format(temparray[i,j], '>11.5f')
        
            f.write(line)
            f.write("\n")
        f.close()

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
