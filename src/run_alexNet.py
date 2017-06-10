import os
import numpy as np
# lrs = np.log(np.logspace(0.000005, 0.0005, 3))
# nes = range(5,20,5)
# ds = np.arange(0.2,1,0.4)
# l2s = np.log(np.logspace(0.005, 0.05, 3))
lrs = np.log(np.logspace(0.000005, 0.0005, 1))
nes = range(5,20,20)
ds = np.arange(0.2,1,1)
l2s = np.log(np.logspace(0.005, 0.05, 1))
params = [(lr,ne,d,l2) for lr in lrs for ne in nes for d in ds for l2 in l2s]
for param in params:
	command = 'python train_alexNet.py --lr ' + str(param[0]) + ' --ne ' + str(param[1]) + ' --d ' + str(param[2]) + ' --l2 ' + str(param[3])
	os.system(command)