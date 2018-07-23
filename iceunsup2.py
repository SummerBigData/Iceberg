# import helping code
import iceFCnn
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
parser.add_argument('iters', help='How many times do we run the base cnn', type=int)
g = parser.parse_args()

print 'You have chosen:', g
print ' '

csvStr = 'submits/subWAvging7-23dn'+str(g.h)+'flip'+str(g.flip)+'iters'+str(g.iters)+'.csv'

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

#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE




# Grab data
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()
unlab, name = grabUnlab()

# Create arrays to store the results
pred = np.zeros(( g.iters, unlab.shape[0] ))
trScore = np.zeros(( g.iters ))
teScore = np.zeros(( g.iters ))

# Run the CNN for g.iters times to collect the predictions from many trials
# The function main automatically loads previous weights if the iteration number, h, and flip are the same
for i in range(g.iters):
	print 'Running iter', i+1
	print ' '
	pred[i], trScore[i], teScore[i] = iceFCnn.main(xtr, ytr, xte, yte, unlab, g.h, g.flip, i)
	print 'Training score for iter', i+1, 'is', trScore[i]
	print 'Testing score for iter', i+1, 'is', teScore[i]
	print ' '

# We use the average prediction, weighted by the test score
avgPred = WeightedAvg(pred, teScore)

# Save the prediction
SavePred(avgPred, name)







