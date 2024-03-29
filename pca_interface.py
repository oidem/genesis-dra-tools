# program pca_interface.py

import sys, math, os
import numpy as np
import pandas as pd
import scipy.stats
import argparse
import pyemma
from matplotlib import pyplot as plt
import time

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'file name you want to reduce the dimensionality')
    parser.add_argument('--o', dest = 'out_name', default = 'umap_result', help = 'root name for output files')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your input data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--scale', dest = 'scaling', default = 'none', help = 'scaling method: none, zscore, max (default: none)')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--ncomponent', dest = 'n_component', default = '-1', help = 'number of dimensions after processing by umap', type = int)
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--debug', dest = 'debug', help = 'debug option', action = 'store_true')

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    scaling     = args.scaling
    skip_output = args.skip_output
    ncomp       = args.n_component
    verbose     = args.verbose
    debug       = args.debug

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
    print('    scaling method: {}'.format(scaling))
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')
    print('    number of dimensions after reduction: {}'.format(ncomp))

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

    ndim = data[0].shape[1]

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)

    if (debug):
        print('')
        print('before scaling, data[0][0,11] = ', data[0][0,11])

    if (scaling == 'zscore'):
        data_temp = np.concatenate(data)
        if (debug):
            print('')
            print('shape of concatenated data: ', data_temp.shape)
        data_scaled = scipy.stats.zscore(data_temp, ddof=1)

        head = 0
        tail = 0
        for i in range(len(data)):
            if (i == 0):
                head = 0
                tail = data[i].shape[0]
            else:
                head += tail
                tail += data[i].shape[0]
            data[i] = data_scaled[head:tail]

    elif (scaling == 'max'):
        data_temp = np.concatenate(data)
        ndata = data_temp.shape[0]

        if (debug):
            print('')
            print('shape of concatenated data: ', data_temp.shape)
        data_max = np.max(data_temp, axis=0)

        if (debug):
            print('')
            print('data_max[11]: ', data_max[11])


        for i in range(ndim):
            if (data_max[i] > 0.0):
                for j in range(ndata):
                    data_temp[j,i] /= data_max[i]

        head = 0
        tail = 0
        for i in range(len(data)):
            if (i == 0):
                head = 0
                tail = data[i].shape[0]
            else:
                head += tail
                tail += data[i].shape[0]

            data[i] = data_temp[head:tail]

    if (debug):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)
        print('')
        print('after scaling, data[0][0,11] = ', data[0][0,11])

    reducer = pyemma.coordinates.pca(data, ncomp)
    embedding = np.concatenate(reducer.get_output())

    ndata = embedding.shape[0]

    if (verbose):
        print('')
        print('shape of concatenated output projection: ', embedding.shape)

    if (ncomp == -1):
        ncomp = embedding.shape[1]

    if (debug):
        print('type of embedding: ', type(embedding))
        print('shape of embedding: ', embedding.shape)

        print('shape of eigvec: ', reducer.eigenvectors.shape)
        print('shape of eigval: ', reducer.eigenvalues.shape)
        print('shape of cumvar: ', reducer.cumvar.shape)

        #print('eigval of 1st IC: ', reducer.eigenvalues[0])
        #print('eigval of 2nd IC: ', reducer.eigenvalues[1])
        print('eigenvector[0, 3]: ', reducer.eigenvectors[0, 3])
        print('eigenvector[2, 3]: ', reducer.eigenvectors[2, 3])

    if (not skip_output):
        fn_png = fn_out + '_plot.png'
        fn_embed = fn_out + '_result.dat'
        fn_vec = fn_out + '_eigvec.dat'
        fn_val = fn_out + '_eigval.dat'

        # output projection
        f = open(fn_embed, "w")
        for i in range(ndata):
            idx = i + 1
            line = format(idx, '>9')
            for j in range(ncomp):
                if (j == 0):
                    line += format(embedding[i,j], '>10.5f')
                else:
                    line += format(embedding[i,j], '>11.5f')

            f.write(line)
            f.write("\n")
        f.close()

        # output eigen vectors
        f = open(fn_vec, "w")
        for i in range(ndim):
            idx = i + 1
            line = format(idx, '>9')
            for j in range(ncomp):
                if (j == 0):
                    line += format(reducer.eigenvectors[i,j], '>10.5f')
                else:
                    line += format(reducer.eigenvectors[i,j], '>11.5f')
        
            f.write(line)
            f.write("\n")
        f.close()

        # output eigen values
        f = open(fn_val, "w")
        for i in range(ncomp):
            idx = i + 1
            line = format(idx, '>9')
            line += format(reducer.eigenvalues[i], '>10.5f')
            line += format(reducer.cumvar[i], '>11.5f')
        
            f.write(line)
            f.write("\n")
        f.close()

        # plot
        plt.xlabel('Component1')
        plt.ylabel('Component2')
        plt.tight_layout()
        plt.scatter(embedding[:,0], embedding[:,1], s = 1)
        plt.savefig(fn_png)

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
