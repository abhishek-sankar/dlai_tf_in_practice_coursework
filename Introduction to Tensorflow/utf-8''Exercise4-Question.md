
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.


```python
import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()
```


```python
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>=DESIRED_ACCURACY):
                self.model.stop_training = True
    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),input_shape=(74,74,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(16,(3,3),input_shape=(37,37,3),activation='softmax'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='softmax'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
        # Your Code Here 
        )
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        # Your Code Here
        '/tmp/h-or-s',
        target_size=(150,150),
        batch_size=8,
        class_mode='binary'
    )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 16,
        epochs = 20,
        verbose = 1,
        callbacks=[callbacks]
    )
    # model fitting
    return history.history['acc'][-1]
```


```python
# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()
```

    Found 80 images belonging to 2 classes.
    Epoch 1/20
    16/16 [==============================] - 3s 200ms/step - loss: 0.6905 - acc: 0.5469
    Epoch 2/20
    16/16 [==============================] - 1s 62ms/step - loss: 0.6792 - acc: 0.6953
    Epoch 3/20
    16/16 [==============================] - 1s 63ms/step - loss: 0.6490 - acc: 0.8906
    Epoch 4/20
    16/16 [==============================] - 1s 62ms/step - loss: 0.6338 - acc: 0.9141
    Epoch 5/20
    16/16 [==============================] - 1s 62ms/step - loss: 0.6232 - acc: 0.9375
    Epoch 6/20
    16/16 [==============================] - 1s 62ms/step - loss: 0.6111 - acc: 0.9531
    Epoch 7/20
    16/16 [==============================] - 1s 57ms/step - loss: 0.6070 - acc: 0.9375
    Epoch 8/20
    16/16 [==============================] - 1s 68ms/step - loss: 0.5989 - acc: 0.9531
    Epoch 9/20
    16/16 [==============================] - 1s 63ms/step - loss: 0.5902 - acc: 0.9688
    Epoch 10/20
    16/16 [==============================] - 1s 62ms/step - loss: 0.5833 - acc: 0.9844
    Epoch 11/20
    16/16 [==============================] - 1s 57ms/step - loss: 0.5745 - acc: 0.9844
    Epoch 12/20
    16/16 [==============================] - 1s 57ms/step - loss: 0.5659 - acc: 1.0000





    1.0




```python
# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook
```


```javascript
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
```


```javascript
%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
```
