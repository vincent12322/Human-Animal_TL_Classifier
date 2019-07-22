# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 22:10:57 2019

@author: eber0
"""

import numpy as np

from keras.models import model_from_json
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


CLASSES = 2
BATCH_SIZE = 10
VAL_BATCH_SIZE = 76


class model(object):

    def __init__(self): 
        #imports the mobilenet model and discards the last 1000 neuron layer.
        base_model = MobileNet(weights='imagenet', include_top=False) 
        
        x = base_model.output
        
        x = GlobalAveragePooling2D()(x)
        
        x = Dense(1024, activation='relu')(x)
        
        x = Dropout(0.35)(x)
        
        x = Dense(1024, activation='relu')(x)
        
        x = Dropout(0.35)(x)
        
        x = Dense(1024, activation='relu')(x)
        
        x = Dense(512, activation='relu')(x)
        
        preds = Dense(CLASSES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=preds)

        for i, layer in enumerate(model.layers):
          print(i, layer.name)
          
        for layer in model.layers:
            layer.trainable=False

        for layer in model.layers[-10:]:
            layer.trainable=True
    
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                         rotation_range=90,
                                         horizontal_flip=True)
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # generate batches of data to train faster with a normalized size
        self.train_generator = train_datagen.flow_from_directory('Data/',
                                                         target_size=(300, 300),
                                                         color_mode='rgb',
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='categorical',
                                                         shuffle=True)
        
        # generate batches of data for validation faster with a normalized size
        self.val_generator = val_datagen.flow_from_directory('Test_Set/',
                                                         target_size=(300, 300),
                                                         color_mode='rgb',
                                                         batch_size=70)

        #Adam optimizer, categorical_crossentropy loss function, measuring accuracy of prediction
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        
    def train(self, epochs):
        step_size_train = self.train_generator.n//self.train_generator.batch_size
        self.model.fit_generator(generator=self.train_generator,
                           steps_per_epoch=step_size_train,
                           epochs=epochs)

    
    def test(self):
        self.val_generator.n//self.val_generator.batch_size
        #Adam optimizer, categorical_crossentropy loss function, measuring accuracy of prediction
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        loss = self.model.evaluate_generator(generator=self.val_generator, steps=1, verbose=1)
        print("Loss = ", loss)
        
    def predict(self, file):
        img_path = 'Picture_Tests/' + file
        
        from PIL import Image
        imshow = Image.open(img_path)
        imshow.show()
        
        img = image.load_img(img_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = self.model.predict(x)
        
        labels = ['Animal', 'Human']
        label = preds.argmax()
        print('Predicted:', labels[label])
        
    def save(self, file, weights):
        model_json = self.model.to_json()
        with open(file, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(weights)
        print("Saved model")
        
    def load(self, file, weights):
        json_file = open(file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights)
        self.model = loaded_model
        