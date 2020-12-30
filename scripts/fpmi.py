import numpy as np
import coalescentmoments as moments


def fpmi(sfs, sfs2):
    fmpi = np.zeros(sfs2.shape)
    n = sfs2.shape[0]
    if n != sfs2.shape[1]:
        print('Warning: 2-SFS matrix not square!')
    for i in range(n):
        for j in range(n):
            if sfs[i] > 0 and sfs[j] > 0 and sfs2[i][j] != 0:
                fmpi[i][j] = np.log(sfs2[i][j] / sfs[i] / sfs[j])
                #fmpi[j][i] = fmpi[i][j]
    return fmpi
