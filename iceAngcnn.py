
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import argparse
#import ijson
# Keras stuff
import keras
from keras.models import load_model, Model #,Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input
from keras.layers import concatenate
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator
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
'''
parser.add_argument("h", help="denoising variable for all colors", type=int)
parser.add_argument("trimsize", help="how many pixels do we trim for data augmentation (even #)?", type=int)
'''

g = parser.parse_args()
g.trimsize=0
g.h=0
g.m = 1604
g.f1 = 75 * 75 * 2
g.f2 = 50
g.f3 = 100
g.f4 = 1
g.imgsize = 75 - g.trimsize
g.epo = 100#300
g.bsize = 200
#saveStr = 'icem'+str(g.m)+'epo'+str(g.epo)+'bsize'+str(g.bsize)+'trimsize'+str(g.trimsize)
saveStr = 'iceEpoAng'+str(g.epo)+'Bsize'+str(g.bsize)#+'h'+str(g.h)+'Trimsize'+str(g.trimsize)
print 'You have chosen:', g
print ' '



#----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE----------DEFINITIONS HERE



def ShowSquare(band1, band2): 
	hspace = np.zeros((75, 5, 3))
	vspace = np.zeros((5, 5*75 + 5*6, 3))
	picAll = vspace
	for i in range(5):
		pici = hspace
		for j in range(5):
			picj = np.zeros((75, 75, 3))
			picj[:,:,0] = Norm(band1[i*5+j,:,:])
			picj[:,:,1] = Norm(band2[i*5+j,:,:])
			pici = np.hstack(( pici, picj, hspace))

		picAll = np.vstack((picAll, pici, vspace))


	imgplot = plt.imshow(picAll, cmap="binary", interpolation='none') 
	plt.show()


def getModel():
	#Building the model
	first_inp = Input(shape=(g.imgsize, g.imgsize, 3))
	#Conv Layer 1
	c1 = Conv2D(64,kernel_size=(3, 3),activation='relu',padding = 'Same')(first_inp)
	p1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c1)
	d1 = Dropout(0.2)(p1)
	#Conv Layer 2
	c2 = Conv2D(128,kernel_size=(3, 3),activation='relu',padding = 'Same')(d1)
	p2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c2)
	d2 = Dropout(0.2)(p2)
	#Conv Layer 3
	c3 = Conv2D(128,kernel_size=(3, 3),activation='relu',padding = 'Same')(d2)
	p3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c3)
	d3 = Dropout(0.2)(p3)
	#Conv Layer 4
	c4 = Conv2D(64,kernel_size=(3, 3),activation='relu',padding = 'Same')(d3)
	p4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c4)
	d4 = Dropout(0.2)(p4)

	#Flatten the data for upcoming dense layers
	f1 = Flatten()(d4)

	#Second Input
	second_inp = Input(shape=(1,))
	#Merge second input with flattened data
	M1 = concatenate([f1, second_inp])

	#Dense Layer 1
	D1 = Dense(512, activation = "relu")(M1)
	#d5 = Dropout(0.2)(D1)
	#Dense Layer 2
	D2 = Dense(256, activation = "relu")(D1)
	#d6 = Dropout(0.2)(D2)
	#Dense Layer 3
	D3 = Dense(64, activation = "relu")(D2)
	#d7 = Dropout(0.2)(D3)
	
	#Sigmoid Layer
	S1 = Dense(1, activation = "sigmoid")(D3)

	gmodel = Model(inputs=([first_inp, second_inp]), outputs=S1)
	
	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
	gmodel.compile(loss='binary_crossentropy',
		optimizer=mypotim,
		metrics=['accuracy'])
	print gmodel.summary()
	return gmodel

# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=8):	
	es = EarlyStopping('val_acc', patience=patience, mode="max")
	msave = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True,save_weights_only=True)
	return [es, msave]

def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)

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
xtr, ytr, atr, xte, yte, ate = iceDataPrep.dataprepAngle()

xtr = decibel_to_linear(xtr)*0.5
xte = decibel_to_linear(xte)*0.5

print xtr[0,0:5, 0:5,1], np.amin(xtr), np.amax(xtr), np.std(xtr)
print atr[0:5], ytr[0:5]
# Add back the angle info as the third chanel
#xtr = iceDataPrep.addAngles(xtr, atr)
#xte = iceDataPrep.addAngles(xte, ate)

# Denoise the images
#xtr = iceDataPrep.denoise(xtr, g.h)
#xte = iceDataPrep.denoise(xte, g.h)
'''
# Denoise the images as an augmentation to the dataset. doubles dataset size
if g.h != 0:
	xtr, ytr = iceDataPrep.augmentDenoise(xtr, ytr, g.h)

# Trim and translate the training set and center trim the test set. quadruples dataset size
if g.trimsize != 0:
	xtr, ytr = iceDataPrep.augmentTranslate(xtr, ytr, g.trimsize, 4)
	xte = iceDataPrep.augmentTranslateCentertrim(xte, g.trimsize)
'''

'''
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
'''


print 'x and y', xtr.shape, ytr.shape
#print 'numIcebergs', sum(y)

# KERAS NEURAL NETWORK



# Get or make the model. Need a different model for each trimsize
if os.path.exists('models/iceModelAng' + str(g.imgsize) ):
	print 'Loading Model'
	model = load_model('models/iceModelAng' + str(g.imgsize) )
else:
	print "Creating Model"
	model = getModel()
	model.save('models/iceModelAng' + str(g.imgsize) )


file_path = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'
callbacks = get_callbacks(filepath=file_path, patience=8)

# Fit the model
#model.fit(x, y, epochs=g.epo, batch_size=g.bsize)
'''
datagen.fit(x)
model.fit_generator(datagen.flow(xtr, ytr, batch_size=g.bsize),
	steps_per_epoch = xtr.shape[0] / (g.bsize+0.0),
	epochs=g.epo,
	verbose = 1,
	validation_data=(xte, yte),
	callbacks=callbacks)
'''

#gmodel=getModel()
model.fit([xtr, atr], ytr,
	batch_size=g.bsize,
	epochs=g.epo,
	verbose=2,
	validation_data=([xte, ate], yte),
	callbacks=callbacks)


# evaluate the model
model.load_weights(file_path)

print 'Accuracy on training data:'
scores = model.evaluate([xtr, atr], ytr)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print 'Accuracy on testing data:'
scores = model.evaluate([xte, ate], yte)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#model.save('models/iceModel' + str(g.imgsize) )
#model.save_weights('weights/' + saveStr)




