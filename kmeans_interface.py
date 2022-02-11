# program kmeans_interface.py

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
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input data you want to classify')
    parser.add_argument('--o', dest = 'out_prefix', default = 'umap_result', help = 'prefix for output files')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your input data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--class', dest = 'n_class', default = '10', help = 'number of classes (default: 10)', type = int)
    parser.add_argument('--ncomponent', dest = 'n_component', default = '-1', help = 'number of dimensions used in clustering', type = int)
    parser.add_argument('--stride', dest = 'stride', default = '1', help = 'skipping step of input data during clustering', type = int)
    parser.add_argument('--iter', dest = 'n_iter', default = '50', help = 'maximum number of iterations in clustering', type = int)
    parser.add_argument('--lag', dest = 'lag_time', default = '10', help = 'lag time step for VAMP2 score', type = int)
    parser.add_argument('--nmode', dest = 'nmode', default = '1', help = 'number of eigenfunctions in VAMP calculation', type = int)
    parser.add_argument('--trial', dest = 'n_trial', default = '5', help = 'number of trials in calculation of vamp', type = int)
    parser.add_argument('--score', dest = 'score_method', default = '', help = 'scoring method for msm consctructed by discretized trajectory: VAMP1 or VAMP2')

    args = parser.parse_args()

    fn_in       = args.in_file
    prefix      = args.out_prefix
    top_file    = args.top_file
    trj         = args.type_trj
    feature     = args.feature
    is_multiple = args.is_multiple
    skip_output = args.skip_output
    verbose     = args.verbose
    nclass      = args.n_class
    ncomp       = args.n_component
    stride      = args.stride
    niter       = args.n_iter
    lag         = args.lag_time
    nmode       = args.nmode
    ntrial      = args.n_trial
    score_method       = args.score_method

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

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

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    output prefix: {}'.format(prefix))
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
    print('kmeans options')
    print('    number of classes: {}'.format(nclass))
    print('    number of dimensions in clustering: {}'.format(ncomp))
    print('    skipping steps in clustering: {}'.format(stride))
    print('    maximum number of iterations: {}'.format(niter))
    print('VAMP options')
    print('    lag time step: {}'.format(lag))
    print('    number of eigenfunctions in VAMP calculation: {}'.format(nmode))
    print('    number of trials of independent VAMP calculation: {}'.format(ntrial))
    print('    scoring method: {}'.format(score_method))

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
            data.append(np.delete(temparray, [0], axis=1))

    data_concatenated = np.concatenate(data)

    if (verbose):
        print('')
        print('shape of each input data:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)
        print('shape of concatenated data: ', data_concatenated.shape)

    ncomp_max = data[0].shape[1]
    if (ncomp == -1):
        if(ncomp_max >= 2):
            ncomp = 2
        else:
            ncomp = 1
    else:
        if(ncomp > ncomp_max):
            print('Warning: the number of dimensions in clustering (specified by you) is larger than the maximum dimensions of input data')
            print('         therefore, number of dimensions is reduced...')
            ncomp = ncomp_max

    temp = copy.deepcopy(data)
    data = []
    for i in range(len(temp)):
        data.append(np.delete(temp[i], slice(ncomp, ncomp_max), axis=1))

    print('')
    print('number of maximum dimensions of input data: ', ncomp_max)
    print('number of dimensions in clustering (after auto-correction): ', ncomp)

    if (verbose):
        print('')
        print('shape of each input data for clustering:')
        for i in range(len(data)):
            print('    ', i, ' ', data[i].shape)

    scores = np.zeros(ntrial)
    score_max = 0.0
    dtraj_out = []
    center_out = np.zeros((nclass, ncomp))
    for i in range(ntrial):
        with pyemma.util.contexts.settings(show_progress_bars=False):

            kmeans = pyemma.coordinates.cluster_kmeans(data, k=nclass, max_iter=niter, stride=stride)
            #if (i == 0):
            #    print('type of dtrajs: ', type(kmeans.dtrajs))
            #    print('dimensions: ', len(kmeans.dtrajs))
            #    print('type of data[0]', type(kmeans.dtrajs[0]))
            #    print('shape of elements: ', kmeans.dtrajs[0].shape)

            msm_kmeans = pyemma.msm.estimate_markov_model(kmeans.dtrajs, lag)

            scores[i] = msm_kmeans.score_cv(kmeans.dtrajs, n=1, score_method=score_method, score_k=nmode)

            if (verbose):
                print('')
                print('trial number: ', i)
                print('    cluster center')
                for j in range(nclass):
                    print('        ', kmeans.clustercenters[j])
                print('    ', score_method,' score: ', scores[i])

            if (scores[i] > score_max):
                dtraj_out = copy.deepcopy(kmeans.dtrajs)
                center_out = kmeans.clustercenters.copy()
                score_max = scores[i]

    print(score_method, ' score in each calculation:')
    for i in range(ntrial):
        line = format(i, '>9')
        line += format(scores[i], '>10.5f')
        print(line)

    print('')
    print(score_method, ' score max: ', score_max)
    print(score_method, ' score mean: ', scores.mean())
    print(score_method, ' score  std: ', scores.std())

    if (not skip_output):

        root, ext = os.path.splitext(fn_in)

        fn_png = root + '_' + prefix + '_kmeans_plot.png'
        fn_center = root + '_' + prefix + '_kmeans_center.dat'

        # plot
        fig, ax = plt.subplots(figsize = (5, 5))
        pyemma.plots.plot_density(*data_concatenated[:, :2].T, ax=ax, cbar=False, alpha=0.3)
        ax.scatter(*center_out[:, :2].T, s=5, c='C1')
        ax.set_xlabel('Component1')
        ax.set_ylabel('Component2')
        fig.tight_layout()
        fig.savefig(fn_png)

        # output centers of clusters
        f = open(fn_center, "w")
        for i in range(nclass):
            idx = i + 1
            line = format(idx, '>9')
            for j in range(ncomp):
                if (j == 0):
                    line += format(center_out[i,j], '>10.5f')
                else:
                    line += format(center_out[i,j], '>11.5f')

            f.write(line)
            f.write("\n")
        f.close()

        # output discretized trajectories
        for i, infile in enumerate(infiles):
            root, ext = os.path.splitext(infile)
            fn_dtraj = root + '_' + prefix + '_kmeans_dtraj.dat'
            f = open(fn_dtraj, "w")
            dtraj = dtraj_out[i]
            for j in range(dtraj.shape[0]):
                idx = j + 1
                line = format(idx, '>9')
                line += format(dtraj[j], '>9')

                f.write(line)
                f.write("\n")
            f.close()

    interval = time.time() - start
    print('processing time : {}s'.format(interval))

    sys.exit()
