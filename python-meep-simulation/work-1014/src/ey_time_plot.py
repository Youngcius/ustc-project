import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

ey_data = pd.read_table(sys.argv[1])


plt.plot(ey_data.iloc[:,0],linewidth=0.8)
plt.title('Ey -- time evolusion')
plt.xlabel('time')
plt.ylabel('Ey')

if len(sys.argv) == 3:
    plt.savefig(sys.argv[2])
else:
    plt.show()

