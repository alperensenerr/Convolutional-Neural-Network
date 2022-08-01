import numpy as np
import pandas as pd
from keras.models import Sequential ,model_from_json
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator


classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. convolution
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Neural network designing
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid')) #there is only 2 diffrent outcome so 1 neuron is enough

# CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['sparse_categorical_accuracy'])

# CNN ve resimler


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')
'''
classifier.fit(training_set,
                         epochs=1,
                         verbose="auto",
                         validation_data = test_set,
                         )

# model saving
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
#X = np.asarray(training_set).astype(np.int_)
#Y = np.array(test_set).astype(np.int_)
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=1)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)

pred[pred > .5] = 1
pred[pred <= .5] = 0

print('prediction gecti')

test_labels = []

for i in range(0,int(len(test_set))):
    test_labels.extend(np.array(test_set[i][1]))
    
#print('test_labels')
#print(test_labels)

dosyaisimleri = test_set.filenames
sonuc = pd.DataFrame()
sonuc['dosyaisimleri']= dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels   
print(sonuc)

cm = confusion_matrix(test_labels, pred)
print (cm)


