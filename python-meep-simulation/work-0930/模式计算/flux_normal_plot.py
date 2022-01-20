import pandas as pd
import matplotlib.pyplot as plt
import sys

flux_data = pd.read_csv(sys.argv[1])
flux0_data = pd.read_csv(sys.argv[2])

flux_normal = flux_data.copy()
flux_normal.iloc[:, 2] = flux_data.iloc[:, 2] / flux0_data.iloc[:, 2]  # normalizing

plt.plot(flux_normal.iloc[:, 1], flux_normal.iloc[:, 2], linewidth=0.7)
plt.scatter(flux_normal.iloc[:, 1], flux_normal.iloc[:, 2], s=7, c='lightblue')
plt.grid()
plt.xlim(0.1, 0.9)
plt.ylim(0, 1.2)
plt.xlabel('Frequency')
plt.ylabel('Transmission')
plt.title('Transmission spectrum (Normalized)')

if len(sys.argv) == 4:
    plt.savefig(sys.argv[3])
else:
    plt.show()
