

import numpy as np
import iceDataPrep
#from skimage.data import camera
from skimage.filters import frangi, hessian, gaussian
from skimage.morphology import disk
from skimage.filters.rank import median
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

def ReadSubmit(string):
	dat = np.genfromtxt(string, delimiter=',')
	labels = dat[1:, 1]
	return labels

def plotCol( x, xstr, y, ystr, label):
	#z_mean, _, _ = encoder.predict(dat, batch_size=g.bsize)
	plt.figure(figsize=(10, 10))
    	plt.scatter(x, y, c=label, s=4)
    	plt.colorbar()
    	plt.xlabel(xstr)
    	plt.ylabel(ystr)
    	#plt.savefig('results/VAEhlCnnAllDrop.png')
    	plt.show()



xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()
ypred = np.genfromtxt('iceBinPredtr0dn10.out', delimiter=','
xtr = iceDataPrep.filterHessian(xtr, 8)
#xtr,_,_ = Norm(xtr, 0, 1)
#yunlab = ReadSubmit('submits/subWAvging7-23dn0flip0iters5.csv')



# Sum over the image dimensions
sumHess = np.sum(xtr[:,:,:,2], axis=1)
sumHess = np.sum(sumHess, axis=1)

plotCol( sumHess, 'Hessian Sum', ytr, 'Y value', np.zeros((ytr.shape)) )


'''
numPics = 15
numChans = 4

imgs = np.zeros((numChans*numPics, 75, 75))

for i in range(numPics):

	gauss = median(xtr[i, :, :, 2], disk(2))
	#markers = ndi.label(markers)[0]


	gauss, _, _ = Norm(gauss, np.amin(xtr[i, :, :, 2]), np.amax(xtr[i, :, :, 1]))

	imgs[i*numChans, :, :] = xtr[i, :, :, 0]
	imgs[i*numChans+1, :, :] = xtr[i, :, :, 1]
	imgs[i*numChans+2, :, :] = xtr[i, :, :, 2]
	imgs[i*numChans+3, :, :] = gauss

picAll = np.zeros((75*numChans, 1))
for i in range(numPics):
	picj = np.zeros((numChans, xtr.shape[1], xtr.shape[1]))
	for j in range(numChans):
		picj[j] = imgs[j+numChans*i]
	pici = np.zeros((0, xtr.shape[1]))
	for j in range(numChans):
		pici = np.concatenate( (pici, picj[j]), axis = 0)
	picAll = np.concatenate( (picAll, pici), axis = 1)
	
imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
plt.show()
'''	










