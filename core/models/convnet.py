import tensorflow.keras as keras

class ConvNet():
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(64,64,1)))
        self.model.add(keras.layers.MaxPool2D((2,2)))
        self.model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
        self.model.add(keras.layers.MaxPool2D((2,2)))
        self.model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
        self.model.add(keras.layers.MaxPool2D((2,2)))
        self.model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
        self.model.add(keras.layers.MaxPool2D((2,2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        def train(self, training_data_gen, validation_data_gen, epochs):
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSProp(lr=1e-4), metrics=['acc'])
            self.model.fit_generator(training_data_gen, epochs=100, validation_data=validation_data_gen)