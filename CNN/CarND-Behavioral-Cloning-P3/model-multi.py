import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


IMAGES_INPUT_SHAPE= (66,200,3) #(128,128,3) #(66,200,3)

local_folder='./Mydata/'
#local_folder='./singledata/'

local_csvfile='driving_log.csv'

# Subfolders where the additional data sets are
data_sets=[
    'curve'
    #'bridge/' #,      # cross bridge
    #'center1/',  # keep on center of lane 1
    #'center2/',   # keep on center of lane 2
    #'center3/',   # keep on center of lane 3
    #'clockwise/',   # clockwise track
    #'curve1/',   # on curve track1
    #'curve2/',   # on curve track2
    #'curve3/',   # on curve track3
    #'leftside/',    # from leftside to center
    #'rightside/'    # from rightside to center
    
    #'1/'
] 

def crop_image(image):
    h=int(image.shape[0])
    w=int(image.shape[1]) 
    #Crop [Y1:Y2, X1:X2] #(0,50)-(w,h-20)
    return image[70:h-25, 0:w] # Top 50px # Bottom 20px

def process_sequential_batch_generator(X, y, batch_size=32):

    N = len(y)
    batches_per_epoch = N // batch_size
    print(' for batch_size=', batch_size, ' need No. for each epoch: ', batches_per_epoch)

    X,y=shuffle(X,y)

    i = 0

    while 1:
        start = i*batch_size
        end = start+batch_size - 1

        if (end >= N):
            print(' end= ', end, 'i= ', i )

        batch_X, batch_y = [], []

        for index in range(start,end):
            if (index>N-1): break
            measurement = y[index]
            image=X[index]

            #if (index==start):
            #    print(image.shape)

            croppred_img = crop_image(image) # (160,320,3)-->(65,320,3)
            #if (index==start):
            #    print(croppred_img.shape)

            resize_img = cv2.resize(croppred_img, (IMAGES_INPUT_SHAPE[1],IMAGES_INPUT_SHAPE[0])) #(65,320,3)->(66,200,3)
            #if (index==start):
            #    print(resize_img.shape)

            batch_X.append(resize_img)
            batch_y.append(measurement)

        i += 1
        if (i == batches_per_epoch-1):
            # reset the index so that we can cycle over the data_frame again
            i = 0

        yield (np.array(batch_X), np.array(batch_y))

def nvidia_model():
    model = Sequential()

    model.add(Lambda(
        lambda x: (x / 255.0) - 0.5, 
        input_shape=(66, 200, 3)
    ))

    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Dropout(.5))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    adam = Adam(lr=0.0001)
    #model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    model.compile(loss="mse", optimizer=adam)
    
    model.summary()
    return model

def save_model(model):    
    #scores = model.evaluate(train_generator, validation_generator, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
     
    # serialize model to JSON
    model_json = model.to_json()
    print(".....Saved model.json to disk......")
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model-weight-curve.h5")
    print("......Saved model weights to disk.........")

images = []
measurements = []
samples = []

correction = 0.2
#zero_steer_pics = 0


def train_track(data, model):
#for data in data_sets:

    # train track by track, save model for each train and restore before next train:
    print('Loading data...', data)


    print(data)
    
    with open(local_folder+data+local_csvfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
            
            '''
            # limit zero steer pics
            if float(line[3])==0.0:
                zero_steer_pics = zero_steer_pics + 1
                if zero_steer_pics > 400:
                    continue
            '''


            # Prepare data and local paths
            for i in range(3): #load 0:center 1:left 2:right

                path=line[i]
                filename=path.split('/')[-1]
                local_path=local_folder+data+'IMG/'+filename
                image=cv2.imread(local_path)
                
                # cut above road part and bottom car part
                #image= crop_image(image)
                
                # Resize all images including validation sets
                #image = cv2.resize(image, (IMAGES_INPUT_SHAPE[1],IMAGES_INPUT_SHAPE[0]))


                # Camera steering correction
                measurement= float(line[3])
                if (i==1):
                    measurement+= correction
                elif(i==2):
                    measurement+= correction

                    
                measurements.append(measurement)
                images.append(image)
                
    # for each sub directory
    print(len(images), len(measurements), len(samples))
    
    augmented_images, augmented_measurements = [], [] 
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
        
    assert len(images)==len(measurements)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    assert len(X_val)==len(y_val)
    print('Training datasets: {}'.format(len(y_train)))
    print('Validation datasets: {}'.format(len(y_val)))

    #return X_train, X_val, y_train, y_val

    print('training for each data track: ...', data)



    train_generator = process_sequential_batch_generator(X_train, y_train, batch_size=32)
    validation_generator = process_sequential_batch_generator(X_val, y_val, batch_size=32)
    
    history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch = len(y_train), 
                                     validation_data = validation_generator, 
                                     nb_val_samples = len(y_val), 
                                     nb_epoch=nb_epoch, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    name = data.split('/')[0]
    plt.savefig(name+'.png')

    #scores = model.evaluate(train_generator, validation_generator, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
     
    # serialize model to JSON
    save_model(model)


nb_epoch = 15 #5

from keras.models import model_from_json

def restore_model():
   # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model-weight-curve.h5")
    print("Loaded model from disk")
     
    adam = Adam(lr=0.0001)
    # evaluate loaded model on test data
    #loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #loaded_model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    loaded_model.compile(optimizer=adam, loss="mse" )

    #score = loaded_model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    return loaded_model


print(' start 1st track .... ')
#model = nvidia_model()
model= restore_model()
train_track('curve/', model)

#print(' start 2nd track .... ')
#model = restore_model()
#train_track('bridge/', model)

'''
model= restore_model()

train_track('center1/', model)

model= restore_model()

train_track('center2/', model)

model= restore_model()
train_track('center3/', model)

model= restore_model()
train_track('clockwise/', model)

model= restore_model()
train_track('curve1/', model)

model= restore_model()
train_track('curve2/', model)

model= restore_model()
train_track('curve3/', model)

model= restore_model()
train_track('leftside/', model)

model= restore_model()
train_track('rightside/', model)
'''

#model.save('model-curve.h5')
model.save('model-curve.h5')