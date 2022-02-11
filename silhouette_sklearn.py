# program tica_interface.py

import sys, math, os
import matplotlib.pyplot as plt
import numpy as np
import pyemma
from pyemma.util.contexts import settings
from sklearn import metrics
import argparse
import time
import copy

if __name__ == '__main__':

    # read commandline variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest = 'in_file', required = True, help = 'input data')
    parser.add_argument('--class', dest = 'class_file', required = True, help = 'result data of clustering for input')
    parser.add_argument('--o', dest = 'out_name', default = 'deviation', help = 'root name of output file')
    parser.add_argument('--top', dest = 'top_file', default = '', help = 'input topology file such as ".pdb".')
    parser.add_argument('--trj', dest = 'type_trj', help = 'whether the datatype of your feature data is trajectory', action = 'store_true')
    parser.add_argument('--feature', dest = 'feature', default = 'all', help = 'feature you want to add to your trajectory input(default: all)')
    parser.add_argument('--multiple', dest = 'is_multiple', help = 'Is your input file is a list of multiple trajcetory?', action = 'store_true')
    parser.add_argument('--stride', dest = 'n_stride', default = '1', help = 'ratio of skipping data over selected data', type = int)
    parser.add_argument('--skip_output', dest = 'skip_output', help = 'whether you want to skip the output of original embedded data', action = 'store_true')
    parser.add_argument('--v', dest = 'verbose', help = 'verbosity', action = 'store_true')
    parser.add_argument('--debug', dest = 'debug', help = 'debug option', action = 'store_true')

    args = parser.parse_args()

    fn_in        = args.in_file
    fn_class     = args.class_file
    fn_out       = args.out_name
    top_file     = args.top_file
    trj          = args.type_trj
    feature      = args.feature
    is_multiple  = args.is_multiple
    nstride      = args.n_stride
    skip_output  = args.skip_output
    verbose      = args.verbose
    debug        = args.debug

    if (trj) and (top_file == ''):
        print('Error: you must input topology file by --top option, if the input datatype is "traj"!!!')
        sys.exit()

    if (nstride <= 0):
        print('Error: --stride must be a positive integer number!!!')
        sys.exit()

    # log output
    print('general options')
    print('    input file: {}'.format(fn_in))
    print('    kmeans result file: {}'.format(fn_class))
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

    infiles = []
    classfiles = []
    if (is_multiple):
        with open(fn_in) as input:
            lines = input.readlines()
            for line in lines:
                infiles.append(line.replace('\n', ''))  # stripping line ending
        with open(fn_class) as input:
            lines = input.readlines()
            for line in lines:
                classfiles.append(line.replace('\n', ''))  # stripping line ending
    else:
        infiles.append(fn_in)
        classfiles.append(fn_class)

    nfile_in = len(infiles)
    nfile_class = len(classfiles)

    if (verbose):
        print('')
        print('number of input files: {}'.format(nfile_in))
        print('number of class files: {}'.format(nfile_class))

    data_temp = []
    class_temp = []

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

        if (nfile_in != 1):
            data_temp = pyemma.coordinates.load(infiles, feat)
        else:
            temparray = pyemma.coordinates.load(infiles, feat)
            data_temp.append(temparray.copy())

        print('type of data: ', type(data_temp))
        print('    number of data points: ', len(data_temp))
        print('type of data[0]', type(data_temp[0]))
        print('    shape of elements: ', data_temp[0].shape)
        print('')
        print('n_atoms: ', feat.topology.n_atoms)
    else:

        for infile in infiles:

            temparray = np.loadtxt(infile)
            data_temp.append(np.delete(temparray, [0], axis = 1))

    for classfile in classfiles:

        temparray = np.loadtxt(classfile)
        class_temp.append(np.delete(temparray, [0], axis = 1))

    data_concatenated = np.concatenate(data_temp)
    class_concatenated = np.concatenate(class_temp)

    ndata_in = int(data_concatenated.shape[0] / nstride)
    ndata_class = int(class_concatenated.shape[0] / nstride)
    ndata = max([ndata_in, ndata_class])

    ndim_in = data_concatenated.shape[1]
    ndim_class = class_concatenated.shape[1]

    templist_in = []
    templist_class = []
    for i in range(ndata):
        idx = i + 1
        if (i < ndata_in):
            templist_in.append(data_concatenated[idx * nstride - 1])
        if (i < ndata_class):
            templist_class.append(class_concatenated[idx * nstride - 1])

    data = np.array(templist_in)
    classes = np.array(templist_class, dtype=int).ravel()


    if (verbose):
        print('')
        print('number of data: ', data.shape[0])
        print('number of classified data points: ', classes.shape[0])
        print('')
        print('dimensionality of data: ', ndim_in)
        print('dimensionality of class: ', ndim_class)

    # debug
    if (debug):
        print('')
        print('shape of data: ', data.shape)
        print('shape of classes: ', classes.shape)
        for i in range(10):
            print(data[i,:])
        print('')
        for i in range(10):
            print(classes[i])
        print(classes[128], data[128,:])
        print(classes[653], data[653,:])
        print(classes[ndata-1], data[ndata-1,:])

    silhouette = metrics.silhouette_samples(data, classes)

    if (debug):
        print(type(silhouette))
        print(silhouette.shape)

    silhouette_mean = np.mean(silhouette)
    silhouette_std = np.std(silhouette, ddof=1) / np.sqrt(ndata)

    # calculate averaged silhouette coefficient for each class
    nclass = np.max(classes) + 1
    silhouette_class = np.zeros(nclass)
    silhouette_std_class = np.zeros(nclass)
    ndata_class = np.zeros(nclass, dtype=int)
    for i in range(nclass):
        templist = []
        for j in range(ndata):
            if classes[j] == i:
                templist.append(silhouette[j])
        temparray = np.array(templist)
        ndata_class[i] = len(templist)
        silhouette_class[i] = np.mean(temparray)
        silhouette_std_class[i] = np.std(temparray, ddof=1) / np.sqrt(ndata_class[i])

    if (verbose):
        print('')
        print('number of classes: ', nclass)
        print('')
        for i in range(nclass):
            print('number of data in class ', i, ': ', ndata_class[i])

    # output
    if (not skip_output):
        fn_silhouette = fn_out + '_silhouette_coeff.dat'
        f = open(fn_silhouette, "w")
        for i in range(ndata):
            idx = i + 1
            line = format(idx, '>9')
            line += format(classes[i], '>9')
            line += format(silhouette[i], '>10.5f')
            f.write(line)
            f.write("\n")
        f.close()

        fn_silhouette_class = fn_out + '_silhouette_class_average.dat'
        f = open(fn_silhouette_class, "w")
        for i in range(nclass):
            line = format(i, '>9')
            line += format(silhouette_class[i], '>10.5f')
            line += format(silhouette_std_class[i], '>10.5f')
            f.write(line)
            f.write("\n")
        f.close()

        print('')
        print('mean of silhouette coefficients for all data', silhouette_mean)
        print(' std of silhouette coefficients for all data', silhouette_std)


    sys.exit()
