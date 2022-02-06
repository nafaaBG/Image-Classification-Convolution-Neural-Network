import os 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
#import pickle
#import cv2_imshow
import cv2
import numpy as np
#import mlflow
#import mlflow.keras

def main():
    project_path = os.getcwd()
    database_path = os.path.join(project_path, 'dataset')    
    base_dir = os.path.join(database_path, 'dataset')
    train_dir = os.path.join(base_dir, 'training_set')
    validation_dir = os.path.join(base_dir, 'test_set')
    
    train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )

    train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, 
                                                    class_mode = 'binary', target_size = (224, 224))
    validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, 
                                                        class_mode = 'binary', target_size = (224, 224))

    base_model = ResNet50(input_shape = (224, 224, 3),include_top = False, weights = 'imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
    #mlflow.keras.autolog()
    results = model.fit(train_generator, validation_data = validation_generator, epochs = 10)
    model.save_weights('vgg16_tf_cat_dog_final_dense2.h5')
    model_json = model.to_json()
    with open("modelVGG_pretrained.json","w") as json_file:
        json_file.write(model_json)

    def predict_(image_path):
        json_file = open('modelVGG_pretrained.json', 'r')
        model_json_c = json_file.read()
        json_file.close()
        model_c = model_from_json(model_json_c)
        model_c.load_weights("vgg16_tf_cat_dog_final_dense.h5")
        opt = SGD(lr=1e-4, momentum=0.9)
        model_c.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))

        preds = model_c.predict(np.expand_dims(image, axis=0))[0]
        if preds==0:
            print("Predicted Label:Cat")
        else:
            print("Predicted Label: Dog")
    #image_path = 'dog.jpg'
    #predict_(image_path)
main()
