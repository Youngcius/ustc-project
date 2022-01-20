import sys
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv(sys.argv[1])  # TE
df2 = pd.read_csv(sys.argv[2])  # TM

plt.figure()

for i in range(6, df1.shape[1]):
    plt.plot(df1[' k1'], df1.iloc[:, i], 'b-', )
for i in range(6, df2.shape[1]):
    plt.plot(df2[' k1'], df2.iloc[:, i], 'r--')
plt.autoscale(tight=True)
plt.grid()
plt.xlabel('k $(2\pi/a)$')
plt.ylabel('Frequancy $(c/a)$')
plt.title('Blue Line: TE; Red Line: TM')

if len(sys.argv) == 4:
    plt.savefig(sys.argv[3])
else:
    plt.show()
