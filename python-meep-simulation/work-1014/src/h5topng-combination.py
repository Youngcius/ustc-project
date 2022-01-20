import os
import sys

eps = sys.argv[1]
data = sys.argv[2]
t = int(sys.argv[3])
cmd = 'h5topng -t 0:{} -R -Zc dkbluered -a yarg -A {} {}'.format(t-1,eps,data)

os.system(cmd)
