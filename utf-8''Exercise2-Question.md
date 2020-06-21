
## Exercise 2
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:
1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

I've started the code for you below -- how would you finish it? 


```python
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
```


```python
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class callbackNow(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
#     print(x_train[0].shape[0])
    # YOUR CODE SHOULD START HERE
    x_train = x_train/255
    x_test = x_test/255
    callback = callbackNow()


    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(units=128,activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE       
        x_train,y_train,epochs=9,callbacks=[callback]
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
```


```python
train_mnist()
```

    Epoch 1/9
    60000/60000 [==============================] - 13s 215us/sample - loss: 0.2588 - acc: 0.9263
    Epoch 2/9
    60000/60000 [==============================] - 12s 200us/sample - loss: 0.1135 - acc: 0.9666
    Epoch 3/9
    60000/60000 [==============================] - 12s 200us/sample - loss: 0.0783 - acc: 0.9759
    Epoch 4/9
    60000/60000 [==============================] - 12s 204us/sample - loss: 0.0589 - acc: 0.9816
    Epoch 5/9
    60000/60000 [==============================] - 12s 205us/sample - loss: 0.0449 - acc: 0.9862
    Epoch 6/9
    60000/60000 [==============================] - 12s 202us/sample - loss: 0.0367 - acc: 0.9883
    Epoch 7/9
    59872/60000 [============================>.] - ETA: 0s - loss: 0.0280 - acc: 0.9914
    Reached 99% accuracy so cancelling training!
    60000/60000 [==============================] - 12s 200us/sample - loss: 0.0280 - acc: 0.9914





    ([0, 1, 2, 3, 4, 5, 6], 0.9914)




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


```python

```
