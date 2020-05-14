import tensorflow as tf#importing tensorflow

class myCallback(tf.keras.callbacks.Callback):  # this allows us to save on time by stopping the training when it reaches a sufficient accuracy
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\ncancelling training")                
      self.model.stop_training = True
 
callbacks = myCallback()

mnist = tf.keras.datasets.mnist # importing and loading the dataset into training and test sets
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train=x_train.reshape(60000, 28, 28, 1) # normalizing and reshaping the training data to make it suitable for the network
x_train  = x_train / 255.0

x_test = x_test.reshape(10000, 28, 28, 1)# normalizing and reshaping the test data to make it suitable for the network
x_test = x_test / 255.0

#now we get to the fun part! This code outlines the architecture of the network.
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])    #I prefer adam to sgd, and the loss is measured using sparse categorical crossentropy

model.summary()
              
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks]) # training the model, takes some time

model.evaluate(x_test, y_test) #tells us how accurate the model is
