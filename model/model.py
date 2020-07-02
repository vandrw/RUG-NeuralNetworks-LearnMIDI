import tensorflow.keras as tfk
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

from input import note_batch_generator

class MidiModel:
    def __init__(self, time_len, batch_len, path = None):
        self.time_len = time_len        
        self.batch_len = batch_len
        if path != None:
            try:
                self.model = tfk.models.load_model(path)
                print("Model successfuly loaded.")
            except:
                print("The model could not be loaded! Creating a new one.")
        else:
            self.model = self.create_model()
        
        #tfk.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)

        self.input_shape = (128,) # TODO Make this dynamic.
            
    def create_model(self):
        model = Sequential()

        model.add(Dense(64, 
                        batch_input_shape=(self.batch_len, self.time_len, 128), 
                        activation="relu", 
                        kernel_regularizer=None))
        model.add(Dropout(0.3))
        model.add(LSTM(80, 
                       stateful=True,
                       kernel_regularizer=None,#regularizers.l2(0.001),
                       recurrent_regularizer=None,#regularizers.l2(0.001), 
                       bias_regularizer=None,
                       activity_regularizer=None))#regularizers.l2(0.001),))
        model.add(Dropout(0.3))
        model.add(Dense(128, 
                        activation="relu",
                        bias_regularizer=None,#regularizers.l2(0.001),
                        kernel_regularizer=None))#regularizers.l2(0.0005)))
        
        model.build(input_shape=(self.batch_len, self.time_len, 128))
        model.compile(loss='mean_squared_error', optimizer='adagrad')
        # model.compile(loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='sgd')

        model.summary()

        return model

    def train(self, epochs, path, max_line_nr):
        loss_update_factor = 0.99
        loss = 1
        for i in range(0, epochs):
            print("Epoch:", i + 1, "/", epochs, "loss:", loss)
            batche_iter = note_batch_generator(path, self.time_len, self.batch_len)
            
            for (line_nr, batches) in batche_iter:
                if line_nr > max_line_nr:
                    break

                self.model.reset_states()
                for (x, y) in batches:
                    batch_loss = self.model.train_on_batch(x, y=y)
                    loss = loss * loss_update_factor + batch_loss * (1.0 - loss_update_factor)
                    print("nr: %5d, loss: %02.4f" % (line_nr, loss), end='\r')

    def save(self, path):
        self.model.save(path)
    
    def predict(self, start_notes):

        pass
