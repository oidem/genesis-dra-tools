# program tica_interface.py

import sys, math, os
import matplotlib.pyplot as plt
import numpy as np
import pyemma
from pyemma.util.contexts import settings
import argparse
import time

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input feature file you want to reduce')
    parser.add_argument('--o', dest = 'out_name', default = 'umap_result', help = 'root name for output files')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your input data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--lag', dest = 'lag_time', default = '10', help = 'lag time step for time-lagged variance-covariance matrix', type = int)
    parser.add_argument('--ncomponent', dest = 'n_component', default = '-1', help = 'number of dimensions kept during the calculation by VAMP', type = int)
    parser.add_argument('--nsplit', dest = 'n_split', default = '10', help = 'number of trials in calculation of vamp', type = int)
    parser.add_argument('--fraction', dest = 'val_frac', default = '0.5', help = 'fraction of random data for validation', type = float)

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    verbose     = args.verbose
    lag         = args.lag_time
    ncomp       = args.n_component
    nsplit      = args.n_split
    frac        = args.val_frac

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output root name: {}'.format(fn_out))
    print('    topology file: {}'.format(top_file))
    if (trj):
        print('    datatype of input: trj')
        print('    feature for VAMP: {}'.format(feature))
    else:
        print('    datatype of input: other')
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('vamp options')
    print('    lag time step: {}'.format(lag))
    print('    number of dimensions kept in VAMP calculation: {}'.format(ncomp))
    print('    number of trials of independent VAMP calculation: {}'.format(nsplit))
    print('    fraction of random data for validation: {}'.format(frac))

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

    if (trj):
        feat = pyemma.coordinates.featurizer(top_file)
        if (feature == 'all'):
            feat.add_all()
        elif (feature == 'backbone_torsions'):
            feat.add_backbone_torsions()
        elif (feature == 'ca_torsions'):
            ca = feat.select_Ca()
            npair = ca.size - 3
            indices = np.zeros((npair, 4), dtype=np.int)
            for i in range(npair):
                for j in range(4):
                    indices[i, j] = ca[i+j]
            feat.add_dihedrals(indices, cossin = True, periodic = False)
        elif (feature == 'distances_ca'):
            feat.add_distances_ca()        

        if (nfile != 1):
            data = pyemma.coordinates.load(infiles, feat)
        else:
            temparray = pyemma.coordinates.load(infiles, feat)
            data.append(temparray.copy())

        print('type of data: ', type(data))
        print('dimensions: ', len(data))
        print('type of data[0]', type(data[0]))
        print('shape of elements: ', data[0].shape)
        print('n_atoms: ', feat.topology.n_atoms)
    else:

        for infile in infiles:

            temparray = np.loadtxt(infile)
            data.append(np.delete(temparray, [0], axis = 1))

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)

    scores = np.zeros(nsplit)
    with pyemma.util.contexts.settings(show_progress_bars=False):

        nval = int(len(data) * frac)
        if (verbose):
            print('')
            print('number of random selection for each split: ', nval)

        for n in range(nsplit):
            ival = np.random.choice(len(data), size=nval, replace=False)

            vamp = pyemma.coordinates.vamp([d for i, d in enumerate(data) if i not in ival], lag =lag, dim = ncomp)
            scores[n] = vamp.score([d for i, d in enumerate(data) if i in ival])

    print('VAMP2 score in each calculation:')
    for i in range(nsplit):
        line = format(i, '>9')
        line += format(scores[i], '>10.5f')
        print(line)

    print('')
    print('VAMP2 score mean: ', scores.mean())
    print('VAMP2 score  std: ', scores.std())

#    reducer = pyemma.coordinates.tica(data, lag, ncomp)
#    embedding = np.concatenate(reducer.get_output())
#
#    if (ncomp == -1):
#        ncomp = embedding.shape[1]
#
#    if (not skip_output):

#        fn_png = fn_out + '_plot.png'
#        fn_embed = fn_out + '_result.dat'
#        fn_vec = fn_out + '_eigvec.dat'
#        fn_val = fn_out + '_eigval.dat'
#
#        # output projection
#        f = open(fn_embed, "w")
#        for i in range(ndata):
#            idx = i + 1
#            line = format(idx, '>9')
#            for j in range(ncomp):
#                if (j == 0):
#                    line += format(embedding[i,j], '>10.5f')
#                else:
#                    line += format(embedding[i,j], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        # output eigen vectors
#        f = open(fn_vec, "w")
#        for i in range(ndim):
#            idx = i + 1
#            line = format(idx, '>9')
#            for j in range(ncomp):
#                if (j == 0):
#                    line += format(reducer.eigenvectors[i,j], '>10.5f')
#                else:
#                    line += format(reducer.eigenvectors[i,j], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        # output eigen values
#        f = open(fn_val, "w")
#        for i in range(ncomp):
#            idx = i + 1
#            line = format(idx, '>9')
#            line += format(reducer.eigenvalues[i], '>10.5f')
#            line += format(reducer.cumvar[i], '>11.5f')
#        
#            f.write(line)
#            f.write("\n")
#        f.close()
#
#        # plot
#        plt.xlabel('Component1')
#        plt.ylabel('Component2')
#        plt.tight_layout()
#        plt.scatter(embedding[:,0], embedding[:,1], s = 1)
#        plt.savefig(fn_png)


    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
