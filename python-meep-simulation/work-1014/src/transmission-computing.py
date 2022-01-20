import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="computing the transmission (ratio of dpwr) ")
parser.add_argument('-h1', type=str, help='name of the hdf5-file 1')
parser.add_argument('-h2', type=str, help='name of the hdf5-file 2')
parser.add_argument('-st', type=int, default=0, help='from where to compute')
args = parser.parse_args()

csvfile1 = args.h1[:-2] + 'csv'
csvfile2 = args.h2[:-2] + 'csv'

os.system('h5totxt' + ' ' + args.h1 + ' ' + '>' + ' ' + csvfile1)
os.system('h5totxt' + ' ' + args.h2 + ' ' + '>' + ' ' + csvfile2)

df1 = pd.read_csv(csvfile1, header=None).iloc[:, args.st:]
df2 = pd.read_csv(csvfile2, header=None).iloc[:, args.st:]
print('shape:', df1.shape)
print('shape:', df2.shape)

os.system('rm ' + csvfile1)
os.system('rm ' + csvfile2)

print('The ratio of dpwr: {}'.format(df1.sum().sum() / df2.sum().sum()))
