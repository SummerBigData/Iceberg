

import numpy as np

def WeightedAvg(pred, weight):
	weightArr = weight.reshape((1, weight.shape[0]))
	
	pred = np.dot(weightArr, pred)
	wAvgPred = np.sum(pred, axis = 0) / ( np.sum(weightArr) + 0.0)
	return wAvgPred



pred = np.array([[0,1,2,3,6,4,3,2,1,2],
		[ 0,1,2,3,6,4,3,2,1,2],
		[ 0,1,2,3,6,4,3,2,1,5]])

weig = np.array([1,2, 3])

wavg = WeightedAvg(pred, weig)
print wavg