# import helping code
import iceF
import iceDataPrep

# Other imports
import argparse
import json
import numpy as np
import pandas as pd
np.random.seed(7)


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES

parser = argparse.ArgumentParser()
parser.add_argument("h", help="denoising variable for all colors", type=int)
parser.add_argument("flip", help="Augmented data using flips? 0 for no, 1 for yes", type=int)
parser.add_argument('iters', help='How many times do we run the base cnn?', type=int)
#parser.add_argument('incl', help='How much pseudolabeled data do we include?', type=int)
g = parser.parse_args()

print 'You have chosen:', g
print ' '

#csvStr = 'submits/subWAvging7-23dn'+str(g.h)+'flip'+str(g.flip)+'iters'+str(g.iters)+'.csv'
#unsupCsvStr = 'submits/subWAvgingUnsup7-23dn'+str(g.h)+'flip'+str(g.flip)+'iters'+str(g.iters)+'incl'+'.csv'
#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


def grabUnlab():
	json_data = open("data/test.json").read()
	dat = json.loads(json_data)

	b1, b2, name, angle  = iceDataPrep.DataSortTest(dat)

	xb1 = b1.reshape((b1.shape[0], 75, 75, 1))
	xb2 = b2.reshape((b1.shape[0], 75, 75, 1))
	xbavg = (xb1 + xb2) / 2.0
	#xbavg = np.zeros(xb1.shape)
	xunlab = np.concatenate((xb1, xb2, xbavg ), axis=3)
	return xunlab, name

def ConvFloattoBin(floatArr):
	binaryArr = np.zeros((floatArr.shape[0])).astype(int)
	for i in range(floatArr.shape[0]):
		if floatArr[i] < 0.5:
			binaryArr[i] = 0
		else:
			binaryArr[i] = 1
	return binaryArr

def WeightedAvg(pred, weight):
	weightArr = weight.reshape((1, weight.shape[0]))
	pred = np.dot(weightArr, pred)
	wAvgPred = np.sum(pred, axis = 0) / ( np.sum(weightArr) + 0.0)
	return wAvgPred

def SavePred(pred, name):
	submission = pd.DataFrame()
	submission['id']= name
	submission['is_iceberg']= pred.reshape((pred.shape[0]))
	submission.to_csv(csvStr, index=False)

def runCnns(xtr, ytr, xte, yte, unlab):
	# Create arrays to store the results
	pred = np.zeros(( g.iters, unlab.shape[0] ))
	# We are storing both the percent and logloss for both training and testing runs
	scores = np.zeros(( g.iters, 2, 2 ))

	# Run the CNN for g.iters times to collect the predictions from many trials
	# The function main automatically loads previous weights if the iteration number, h, and flip are the same
	for i in range(g.iters):
		print 'Running iter', i+1
		print ' '
		pred[i], scores[i] = iceF.cnn(xtr, ytr, xte, yte, unlab, g.h, g.flip, i)
		print 'Training percent for iter', i+1, 'is', scores[i, 0, 1]*100, 'with log loss', scores[i, 0, 0]
		print 'Testing percent for iter', i+1, 'is', scores[i, 1, 1]*100, 'with log loss', scores[i, 1, 0]
		print ' '

	# We use the average prediction, weighted by the test score
	avgPred = WeightedAvg(pred, 1.0/scores[:, 1, 0])
	return avgPred

def runAE(xtr, ytr, xte, yte, unlab, flip):
	
	xtrPred, unlabPred, scores = iceF.autoencoder(xtr, ytr, xte, yte, unlab, flip)

	print score
	return xtrPred, unlabPred

def predictAE(xtr):
	import keras
	from keras.models import load_model
	from keras.optimizers import Adam

	epo = 70
	bsize = 100
	imgsize = 75
	saveStr = 'iceunsup2/AEepo'+str(epo)+'Bsize'+str(bsize)+'flip'+str(g.flip)
	saveStr = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'

	autoencoder = load_model('models/iceAE' + str(imgsize) )
	encoder = load_model('models/iceEncoder' + str(imgsize) )

	optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	autoencoder.compile(optimizer=optimizer, loss='mse', metrics=["accuracy"])
	autoencoder.load_weights(saveStr)

	encPred = encoder.predict(xtr)
	aePred = autoencoder.predict(xtr)

	return encPred, aePred

def ShowSquare(x): 
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg

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


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE




# Grab data
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()
unlab, name = grabUnlab()

xtrT, xtrTT = iceDataPrep.augmentPCA(xtr, 25)
xteT, xteTT = iceDataPrep.augmentPCA(xte, 25)
unlabT, unlabTT = iceDataPrep.augmentPCA(unlab, 25)

#avgPred = runCnns(xtrTT, ytr, xteTT, yte, unlabTT)

prediction, results = iceF.cnnPCA(xtr, xtrT, ytr, xte, xteT, yte, unlab, 0)




#xtrPred, unlabPred = runAE(xtr, ytr, xte, yte, unlab, g.flip)





'''
print ((xtrT - xtr) ** 2).mean(axis=None)


plotimg = np.zeros((36, 75, 75, 3))
for i in range(18):
	plotimg[i*2] = xtr[i]
	plotimg[i*2+1] = xtrT[i]

ShowSquare(plotimg)
'''

# Save the prediction
#SavePred(avgPred, name)

print 'Done'




