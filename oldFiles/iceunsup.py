
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import argparse
import json
# Keras stuff
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os.path # To check if a file exists
import iceDataPrep

# Print the exact architecture being used
import sys
print ' '
print(sys.version)
print ' '

#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
#parser.add_argument("m", help="Number of Datapoints, up to 1604", type=int)
parser.add_argument("h", help="denoising variable for all colors", type=int)
parser.add_argument("trimsize", help="how many pixels do we trim for data augmentation (even #)?", type=int)
parser.add_argument("pseuSize", help="How many pseudo images should be added?", type=int)
g = parser.parse_args()
g.m = 1604
g.f1 = 75 * 75 * 2
g.f2 = 50
g.f3 = 100
g.f4 = 1
g.imgsize = 75 - g.trimsize
g.epo = 70#300
g.bsize = 100
#saveStr = 'icem'+str(g.m)+'epo'+str(g.epo)+'bsize'+str(g.bsize)+'trimsize'+str(g.trimsize)
saveStr = 'iceIter1Epo'+str(g.epo)+'Bsize'+str(g.bsize)+'h'+str(g.h)+'Trimsize'+str(g.trimsize) + 'psize' + str(g.pseuSize)
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE


# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=8):	
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True,save_weights_only=True)
	return [es, msave]


#----------STARTS HERE----------STARTS HERE----------STARTS HERE----------STARTS HERE


'''
train = pd.read_json("data/train.json")	# 1604 x 5


filename = "data/test.json"
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)

print columns.shape


# Read out the data in the two bands, as 1604 x 75 x 75 arrays
TRb1, TRb2, TRname, TRlabel, TRangle, TRonlyAngle = DataSort(train)
'''

# DATA PREP

# Grab the training (tr) and testing (te) data and labels
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprep()

# Grab the unlabeled data
json_data = open("data/test.json").read()
dat = json.loads(json_data)
b1, b2, name, angle  = iceDataPrep.DataSortTest(dat)
# Reshape it
xb1 = b1.reshape((b1.shape[0], 75, 75, 1))
xb2 = b2.reshape((b1.shape[0], 75, 75, 1))
xbavg = (xb1 + xb2) / 2.0
#xbavg = np.zeros(xb1.shape)
xtrpseudo = np.concatenate((xb1, xb2, xbavg ), axis=3)

# Grab the pseudolabels
ytrpseudo = np.genfromtxt('iceBinPredtr'+str(g.trimsize)+'dn'+str(g.h)+'.out',dtype=float,delimiter=',')


# Now stick together 250 images from the unlabeled data with the labels calculated
xtrNew = np.concatenate((xtr, xtrpseudo[0:g.pseuSize]), axis=0)
ytrNew = np.concatenate((ytr, ytrpseudo[0:g.pseuSize]), axis=0)
# Shuffle it for good measure
xtrNew, ytrNew = iceDataPrep.shuffleData(xtrNew, ytrNew)




'''
# Denoise the images as an augmentation to the dataset. doubles dataset size
if g.h != 0:
	xtr, ytr = iceDataPrep.augmentDenoise(xtr, ytr, g.h)

# Trim and translate the training set and center trim the test set. quadruples dataset size
if g.trimsize != 0:
	xtr, ytr = iceDataPrep.augmentTranslate(xtr, ytr, g.trimsize, 4)
	xte = iceDataPrep.augmentTranslateCentertrim(xte, g.trimsize)
'''

print 'x and y', xtrNew.shape, ytrNew.shape


# KERAS NEURAL NETWORK



# Get or make the model. Need a different model for each trimsize
# Load the model and weights and make the predictions
model = load_model('models/iceModel'+str(g.imgsize) )



file_path = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'
callbacks = get_callbacks(filepath=file_path, patience=8)

# Fit the model


#gmodel=getModel()
model.fit(xtrNew, ytrNew,
	batch_size=g.bsize,
	epochs=g.epo,
	verbose=2,
	validation_data=(xte, yte),
	callbacks=callbacks)


# evaluate the model
model.load_weights(file_path)

print 'Accuracy on training data:'
scores = model.evaluate(xtrNew, ytrNew)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print 'Accuracy on testing data:'
scores = model.evaluate(xte, yte)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save('models/iceModel' + str(g.imgsize) )
#model.save_weights('weights/' + saveStr)




