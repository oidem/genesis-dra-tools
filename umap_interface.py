# program umap_interface.py

import sys, math, os
import numpy as np
import pandas as pd
import argparse
import umap
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
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--ncomponent', dest = 'n_component', default = '2', help = 'number of dimensions after processing by umap', type = int)
    parser.add_argument('--neighbor', dest = 'n_neighbors', default = '15', help = 'n_neighbors of sklearn, which controls the balance between local and global structure', type = int)
    parser.add_argument('--min_dist', dest = 'min_dist', default = '0.1', help = 'min_dist of sklearn, which controls the tightness of the packed data points', type = float)
    parser.add_argument('--spread', dest = 'spread', default = '1.0', help = 'effective scale of embedded points', type = float)
    parser.add_argument('--metric', dest = 'metric', default = 'euclidean', help = 'metric of sklearn, which defines the distance among data points')
    parser.add_argument('--split', dest = 'split_data', help = 'whether you want to split the data to reduce the computational cost', action = 'store_true')
    parser.add_argument('--seed', dest = 'random_seed', default = '-1', help = 'random seed for initialization of k-means analysis', type = int)
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')

    args = parser.parse_args()

    fn_in       = args.in_file
    fn_out      = args.out_name
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    ncomp       = args.n_component
    neighbors   = args.n_neighbors
    mindist     = args.min_dist
    spread_     = args.spread
    metric_     = args.metric
    split       = args.split_data
    seed        = args.random_seed
    verbose     = args.verbose

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
    print('umap options')
    print('    number of dimensions after reduction: {}'.format(ncomp))
    print('    number of neighbors: {}'.format(neighbors))
    print('    minimum distance for embedding: {}'.format(mindist))
    print('    effective scale of embedded points: {}'.format(spread_))
    print('    metric for distance calculation: {}'.format(metric_))
    print('    random seed: {}'.format(seed))
    print('expert option: for embedding additional data')

    if (seed < 0):
        seed = None

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

    data_concatenated = np.concatenate(data)

    ndata = data_concatenated.shape[0]

    nlearn = ndata
    if (split):
        nlearn = int(ndata / 2)

    nembed = nlearn - ndata

    data_learn = np.zeros((nlearn, ndim))
    data_embed = np.zeros((nembed, ndim))

#    if (split):
#        data_learn = 



#    if (datatype == 'weight'):
#        sum_data = np.sum(data, axis = 1)
#        for i in range(ndata):
#            for j in range(ndim):
#                data[i, j] /= sum_data[i]

    reducer = umap.UMAP(n_neighbors = neighbors, n_components = ncomp, min_dist = mindist, spread = spread_, metric = metric_, random_state = seed)

    embedding = reducer.fit(data_concatenated)

    if (verbose):
        print('')
        print('shape of concatenated output projection: ', embedding.embedding_.shape)

    if (not skip_output):
        fn_png = fn_out + '_plot.png'
        fn_embed = fn_out + '_result.dat'

        # output result
        f = open(fn_embed, "w")
        for i in range(ndata):
            idx = i + 1
            line = format(i, '>9')
            for j in range(ncomp):
                if (j == 0):
                    line += format(embedding.embedding_[i,j], '>10.5f')
                else:
                    line += format(embedding.embedding_[i,j], '>11.5f')
        
            f.write(line)
            f.write("\n")
        f.close()

        # plot
        plt.xlabel('Component1')
        plt.ylabel('Component2')
        plt.tight_layout()
        plt.scatter(embedding.embedding_[:,0], embedding.embedding_[:,1], s = 1)
        plt.savefig(fn_png)

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
