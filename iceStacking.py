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
#parser.add_argument("h", help="denoising variable for all colors", type=int)
#parser.add_argument("flip", help="Augmented data using flips? 0 for no, 1 for yes", type=int)
#parser.add_argument('iters', help='How many times do we run the base cnn?', type=int)
#parser.add_argument('incl', help='How much pseudolabeled data do we include?', type=int)
g = parser.parse_args()
#g.flip = 0
print 'You have chosen:', g
print ' '

csvStr = 'submits/subWAvging7-23dn0flip0iters5.csv'
saveStr = 'submits/subPullPred98WAvging7-23dn0flip0iters5.csv'

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
	submission.to_csv(saveStr, index=False)

def ReadSubmit(string):
	dat = np.genfromtxt(string, delimiter=',')
	labels = dat[1:, 1]
	return labels

def PushAcc(bestPred, threshold):
	pushPred = np.zeros(bestPred.shape).astype(float)

	for i in range(bestPred.shape[0]):
		if bestPred[i] > threshold:
			pushPred[i] = 1
		elif bestPred[i] < 1 -threshold:
			pushPred[i] = 0
		else:
			pushPred[i] = bestPred[i]
	return pushPred

def PullAcc(bestPred, threshold):
	pullPred = np.zeros(bestPred.shape).astype(float)

	for i in range(bestPred.shape[0]):
		if bestPred[i] > threshold:
			pullPred[i] = threshold
		elif bestPred[i] < 1 -threshold:
			pullPred[i] = 1 - threshold
		else:
			pullPred[i] = bestPred[i]
	return pullPred



#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE




# Grab data
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()
unlab, name = grabUnlab()

bestPred = ReadSubmit(csvStr)

pullPred = PullAcc(bestPred, 0.98)




SavePred(pullPred, name)

print 'Done'




