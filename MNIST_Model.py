
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\ncancelling training")
      self.model.stop_training = True
callbacks = myCallback()   
            
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
            
x_train=x_train.reshape(60000, 28, 28, 1)
x_train  = x_train / 255.0
            
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',kernel_initializer='he_uniform',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
  tf.keras.layers.Dense(10, activation='softmax')
])
  
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
  
model.evaluate(test_images, test_labels)
