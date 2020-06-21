
## Exercise 3
In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.

I've started the code for you -- you need to finish it!

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"



```python
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
```


```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```


```python
# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.998):
                self.model.stop_training = True
    callback = myCallback()
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000,28,28,1)
    training_images = training_images/255
    test_images = test_images.reshape(10000,28,28,1)
    test_images = test_images/255
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
            tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(10,activation='softmax')
        
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images,training_labels,epochs=20,callbacks=[callback]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]


```


```python
_, _ = train_mnist_conv()
```

    Epoch 1/20
    60000/60000 [==============================] - 17s 289us/sample - loss: 0.1571 - acc: 0.9515
    Epoch 2/20
    60000/60000 [==============================] - 17s 282us/sample - loss: 0.0488 - acc: 0.9850
    Epoch 3/20
    60000/60000 [==============================] - 17s 277us/sample - loss: 0.0353 - acc: 0.9890
    Epoch 4/20
    60000/60000 [==============================] - 17s 287us/sample - loss: 0.0264 - acc: 0.9921
    Epoch 5/20
    60000/60000 [==============================] - 17s 281us/sample - loss: 0.0206 - acc: 0.9936
    Epoch 6/20
    60000/60000 [==============================] - 16s 267us/sample - loss: 0.0162 - acc: 0.9949
    Epoch 7/20
    60000/60000 [==============================] - 17s 280us/sample - loss: 0.0132 - acc: 0.9957
    Epoch 8/20
    60000/60000 [==============================] - 17s 277us/sample - loss: 0.0105 - acc: 0.9962
    Epoch 9/20
    60000/60000 [==============================] - 16s 275us/sample - loss: 0.0093 - acc: 0.9969
    Epoch 10/20
    60000/60000 [==============================] - 17s 282us/sample - loss: 0.0085 - acc: 0.9973
    Epoch 11/20
    60000/60000 [==============================] - 17s 287us/sample - loss: 0.0069 - acc: 0.9976
    Epoch 12/20
    60000/60000 [==============================] - 18s 293us/sample - loss: 0.0070 - acc: 0.9976
    Epoch 13/20
    60000/60000 [==============================] - 17s 290us/sample - loss: 0.0053 - acc: 0.9983



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
