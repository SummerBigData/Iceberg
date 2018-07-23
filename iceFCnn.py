
import numpy as np

# Keras stuff
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import plot_model
import os.path # To check if a file exists
import iceDataPrep


#---------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES----------GLOBAL VARIABLES


# fix random seed for reproducibility
np.random.seed(7)




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
	gmodel=Sequential()
	#Conv Layer 1
	gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(g.imgsize, g.imgsize, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	
	#Conv Layer 2
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Conv Layer 3
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	
	#Conv Layer 4
	gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Flatten the data for upcoming dense layers
	gmodel.add(Flatten())
	
	#Dense Layers
	gmodel.add(Dense(512))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))

	#Dense Layer 2
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))
	
	#Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('sigmoid'))

	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='binary_crossentropy',
		optimizer=mypotim,
		metrics=['accuracy'])
	gmodel.summary()
	return gmodel

# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=8):	
	es = EarlyStopping('val_acc', patience=patience, mode="max")
	msave = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True,save_weights_only=True)
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
def main(xtr, ytr, xte, yte, unlab, h, flip, ind):
	# Use a seed based on the index
	np.random.seed(ind)

	epo = 70
	bsize = 100
	imgsize = 75
	saveStr = 'iceunsup2/Epo'+str(epo)+'Bsize'+str(bsize)+'h'+str(h)+'flip'+str(flip)+'ind'+str(ind)
	saveStr = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'

	# Denoise the images as an augmentation to the dataset. doubles dataset size
	if h != 0:
		xtr, ytr = iceDataPrep.augmentDenoise(xtr, ytr, g.h)

	# Trim and translate the training set and center trim the test set. quadruples dataset size
	if flip != 0:
		xtr, ytr = iceDataPrep.augmentFlip(xtr, ytr)

	# KERAS NEURAL NETWORK

	# Get or make the model. Need a different model for each trimsize
	if os.path.exists('models/iceModel' + str(imgsize) ):
		model = load_model('models/iceModel' + str(imgsize) )
	else:
		model = getModel()
	
	# Get or do the run. No need to run things more than necessary, right?
	if os.path.exists(saveStr):
		print 'Pulling index', ind, 'from previous runs'
		model.load_weights(saveStr)
	
	else:
		callbacks = get_callbacks(filepath=saveStr, patience=12)
		# Fit the model
		model.fit(xtr, ytr,
			batch_size=bsize,
			epochs=epo,
			verbose=2,
			validation_data=(xte, yte),
			callbacks=callbacks)

	# evaluate the model
	model.load_weights(saveStr)

	# Calculate the scores on the training and testing data
	results = np.zeros((2, 2))
	# Training
	scores = model.evaluate(xtr, ytr, verbose=0)
	results[0, 0] = scores[0]
	results[0, 1] = scores[1]
	# Testing
	scores = model.evaluate(xte, yte, verbose=0)
	results[1, 0] = scores[0]
	results[1, 1] = scores[1]

	prediction = model.predict(unlab)

	return prediction.flatten(), results
#model.save('models/iceModel' + str(g.imgsize) )
#model.save_weights('weights/' + saveStr)




