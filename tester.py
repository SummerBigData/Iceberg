

import numpy as np
import iceDataPrep
#from skimage.data import camera
from skimage.filters import frangi, hessian

import matplotlib.pyplot as plt


def ShowSquare(x): 
	#import matplotlib.pyplot as plt

	hspace = np.zeros((75, 5, 3))
	vspace = np.zeros((5, 6*75 + 5*7, 3))
	picAll = vspace
	for i in range(6):
		pici = hspace
		for j in range(6):
			picj = x[i*6+j, :, :, :]
			picj, Min, Max = Norm(picj, 0, 1)
			pici = np.hstack(( pici, picj, hspace))

		picAll = np.vstack((picAll, pici, vspace))


	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
	plt.show()


def Norm(mat, nMin, nMax):
	# Calculate the old min, max and convert to float values
	Min = np.amin(mat).astype(float)
	Max = np.amax(mat).astype(float)
	nMin = nMin+0.0
	nMax = nMax+0.0
	# Calculate the new normalized matrix
	normMat = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return normMat, Min, Max



xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()

xtr = iceDataPrep.filterHessian(xtr)




imgs = np.zeros((30, 75, 75))

for i in range(10):

	imgs[i*3, :, :] = xtr[i, :, :, 0]
	imgs[i*3+1, :, :] = xtr[i, :, :, 1]
	imgs[i*3+2, :, :] = xtr[i, :, :, 2]

picAll = np.zeros((75*3, 1))
for i in range(10):
	pici = np.concatenate( (imgs[0+3*i], imgs[1+3*i], imgs[2+3*i]), axis = 0)
	picAll = np.concatenate( (picAll, pici), axis = 1)
	
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.show()
		










