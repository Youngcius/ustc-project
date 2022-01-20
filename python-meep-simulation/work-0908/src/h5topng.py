import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse


parser = argparse.ArgumentParser(
    description="plot line figure based on one-dimension hdf5-format file ")
parser.add_argument('-h5', type=str, help='name of the hdf5-file')
parser.add_argument('-plot', type=str,
                    help='name of output figure name', default=None)

args = parser.parse_args()

h5file = args.h5
txtfile = args.h5[:-2]+'txt'
if args.plot == None:
    pngfile = args.h5[:-2]+'png'
else:
    pngfile = args.plot

os.system('/usr/bin/h5totxt' + ' ' + h5file + ' ' + '>' + ' ' + txtfile)

df = pd.read_csv(txtfile,header=None)
os.system('rm '+txtfile)
times = np.arange(df.size)*0.1
init_time = 0
# init_time = np.where(df!=0)[0][0]
# print(init_time)
plt.plot(times[init_time:],df.iloc[:, 0][init_time:],linewidth=0.2)
plt.xlabel('time/s')
plt.title(pngfile)
plt.title(pngfile)
# plt.show()
plt.savefig(pngfile)
