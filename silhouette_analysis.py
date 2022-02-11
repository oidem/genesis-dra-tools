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
    parser.add_argument('--i1', dest = 'in_file1', required = True, help = 'input data 1')
    parser.add_argument('--i2', dest = 'in_file2', required = True, help = 'input data 2')
    parser.add_argument('--o', dest = 'out_name', default = 'deviation', help = 'root name of output file')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your feature data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--stride', dest = 'n_stride', default = '1', help = 'ratio of skipping data over selected data', type = int)
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')

    args = parser.parse_args()

    fn_in1       = args.in_file1
    fn_in2       = args.in_file2
    fn_out       = args.out_name
    top_file     = args.top_file
    trj          = args.type_trj
    feature      = args.feature
    is_multiple  = args.is_multiple
    nstride      = args.n_stride
    skip_output  = args.skip_output
    verbose      = args.verbose

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

    if (nstride <= 0):
        print('Error: --stride must be a positive integer number!!!')
        sys.exit()

    # log output
    print('general options')
    print('    input file 1: {}'.format(fn_in1))
    print('    input file 2: {}'.format(fn_in2))
    print('    output fn_out: {}'.format(fn_out))
    print('    topology file: {}'.format(top_file))
    if (trj):
        print('    datatype of input: trj')
        print('    feature: {}'.format(feature))
    else:
        print('    datatype of input: other')
    print('    width of stride: {}'.format(nstride))
    if (skip_output):
        print('    skip output section?: yes')
    else:
        print('    skip output section?: no')

    start = time.time()

    infiles1 = []
    infiles2 = []
    if (is_multiple):
        with open(fn_in1) as input:
            lines = input.readlines()
            for line in lines:
                infiles1.append(line.replace('\n', ''))  # stripping line ending
        with open(fn_in2) as input:
            lines = input.readlines()
            for line in lines:
                infiles2.append(line.replace('\n', ''))  # stripping line ending
    else:
        infiles1.append(fn_in1)
        infiles2.append(fn_in2)

    nfile1 = len(infiles1)
    nfile2 = len(infiles2)

    if (verbose):
        print('')
        print('number of input files 1: {}'.format(nfile1))
        print('number of input files 2: {}'.format(nfile2))

    data1_temp = []
    data2_temp = []

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

        if (nfile1 != 1):
            data1_temp = pyemma.coordinates.load(infiles1, feat)
            data2_temp = pyemma.coordinates.load(infiles2, feat)
        else:
            temparray = pyemma.coordinates.load(infiles1, feat)
            data1_temp.append(temparray.copy())
            temparray = pyemma.coordinates.load(infiles2, feat)
            data2_temp.append(temparray.copy())

        print('type of data 1: ', type(data1))
        print('    number of data points: ', len(data1))
        print('type of data1[0]', type(data1[0]))
        print('    shape of elements: ', data1[0].shape)
        print('')
        print('type of data 2: ', type(data2))
        print('    number of data points: ', len(data2))
        print('type of data2[0]', type(data2[0]))
        print('    shape of elements: ', data2[0].shape)
        print('')
        print('n_atoms: ', feat.topology.n_atoms)
    else:

        for infile in infiles1:

            temparray = np.loadtxt(infile)
            data1_temp.append(np.delete(temparray, [0], axis = 1))

        for infile in infiles2:

            temparray = np.loadtxt(infile)
            data2_temp.append(np.delete(temparray, [0], axis = 1))

    data1_concatenated = np.concatenate(data1_temp)
    data2_concatenated = np.concatenate(data2_temp)

    ndata1 = int(data1_concatenated.shape[0] / nstride)
    ndata2 = int(data2_concatenated.shape[0] / nstride)
    ndata = max([ndata1, ndata2])

    ndim1 = data1_concatenated.shape[1]
    ndim2 = data2_concatenated.shape[1]

    templist1 = []
    templist2 = []
    for i in range(ndata):
        idx = i + 1
        if (i < ndata1):
            templist1.append(data1_concatenated[idx * nstride - 1])
        if (i < ndata2):
            templist2.append(data2_concatenated[idx * nstride - 1])

    data1 = np.array(templist1)
    data2 = np.array(templist2)

    if (verbose):
        print('')
        print('number of data1: ', data1.shape[0])
        print('number of data2: ', data2.shape[0])
        print('')
        print('dimensionality of data1: ', ndim1)
        print('dimensionality of data2: ', ndim2)

    if (ndim1 != ndim2):
        print('')
        print('Error: the dimensionality sizes of data 1 and 2 are different. We cannot compute distance between these data points.')
        print('Exitting...')
        sys.exit()

    s12 = np.zeros(ndata1)
    s21 = np.zeros(ndata2)    

    ndata = max([ndata1, ndata2])

    for i in range(ndata):
        temp1_i = np.zeros(ndim1)
        temp2_i = np.zeros(ndim2)

        a12 = 0.0
        b12 = 0.0

        a21 = 0.0
        b21 = 0.0

        if (i <= ndata1 - 1):
            temp1_i = data1[i]
        if (i <= ndata2 - 1):
            temp2_i = data2[i]

        for j in range(ndata):
            temp1_j = np.zeros(ndim1)
            temp2_j = np.zeros(ndim2)

            if (j <= ndata1 - 1):
                temp1_j = data1[j]
                if (i <= ndata1 - 1):
                    diff = temp1_i - temp1_j
                    a12 += np.linalg.norm(diff)
                if (i <= ndata2 - 1):
                    diff = temp2_i - temp1_j
                    b21 += np.linalg.norm(diff)
            if (j <= ndata2 - 1):
                temp2_j = data2[j]
                if (i <= ndata1 - 1):
                    diff = temp1_i - temp2_j
                    b12 += np.linalg.norm(diff)
                if (i <= ndata2 - 1):
                    diff = temp2_i - temp2_j
                    a21 += np.linalg.norm(diff)

        a12 /= ndata1 - 1
        b12 /= ndata2

        a21 /= ndata2 - 1
        b21 /= ndata1

#        print('')
#        print('a12: ', a12)
#        print('b12: ', b12)
#        print('a21: ', a21)
#        print('b21: ', b21)

        if (i <= ndata1 - 1):
            s12[i] = (b12 - a12) / max([a12, b12])
        if (i <= ndata2 - 1):
            s21[i] = (b21 - a21) / max([a21, b21])

    s12_mean = np.mean(s12)
    s12_std  = np.std(s12, ddof=1) / np.sqrt(ndata1)

    s21_mean = np.mean(s21)
    s21_std  = np.std(s21, ddof=1) / np.sqrt(ndata2)

    print('')
    print('s12')
    print('    mean: ', s12_mean)
    print('     std: ', s12_std)
    print('s21')
    print('    mean: ', s21_mean)
    print('     std: ', s21_std)


    if (not skip_output):

        fn_silhouette1 = fn_out + '_silhouette_coeff_1.dat'
        fn_silhouette2 = fn_out + '_silhouette_coeff_2.dat'

        f = open(fn_silhouette1, "w")
        for i in range(len(s12)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(s12[i], '>10.5f')
            f.write(line)
            f.write("\n")
        f.close()

        f = open(fn_silhouette2, "w")
        for i in range(len(s21)):
            idx = i + 1
            line = format(idx, '>9')
            line += format(s21[i], '>10.5f')
            f.write(line)
            f.write("\n")
        f.close()

    sys.exit()
