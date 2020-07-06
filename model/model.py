import tensorflow.keras as tfk
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import deque
import numpy as np
import random

class MidiModel:
    def __init__(self, cutoff, time_len, batch_len, path = None):
        self.time_len = time_len        
        self.batch_len = batch_len
        self.cutoff = cutoff
            
        if path != None:
            try:
                self.model = tfk.models.load_model(path)
                print("Model successfuly loaded.")
            except:
                print("The model could not be loaded! Creating a new one.")
        else:
            self.model = self.create_model(self.batch_len)
        
        self.model.summary()
        #tfk.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)

    def create_model(self, batch_len):
        model = Sequential()

        model.add(Dense(64, 
                        batch_input_shape=(batch_len, self.time_len, 128), 
                        activation="sigmoid", 
                        kernel_regularizer=None))
        # model.add(Dropout(0.3))
        model.add(LSTM(128, 
                       stateful=True,
                       kernel_regularizer=None,#regularizers.l2(0.001),
                       recurrent_regularizer=None,#regularizers.l2(0.001), 
                       bias_regularizer=None,
                       activity_regularizer=None))#regularizers.l2(0.001),))
        # model.add(Dropout(0.1))
        model.add(Dense(128, 
                        activation="sigmoid",
                        bias_regularizer=None,#regularizers.l2(0.001),
                        kernel_regularizer=None))#regularizers.l2(0.0005)))
        
        model.build(input_shape=(batch_len, self.time_len, 128))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        # model.compile(loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='sgd')

        return model

    def train(self, epochs, songs):
        for i in range(0, epochs):
            random.shuffle(songs)
            batche_iter = batch_generator(songs, self.time_len, self.batch_len)
            
            loss_sum = 0
            loss_sum_count = 0
            for batches in batche_iter:
                self.model.reset_states()
                for (x, y) in batches:
                    loss_sum += self.model.train_on_batch(x, y=y)
                    loss_sum_count += 1
                    print("Epoch: %d/%d Loss: %02.4f" % (i + 1, epochs, loss_sum / loss_sum_count), end='          \r')
            print()

    def save(self, path):
        self.model.save(path)
    
    def predict(self, start_notes):
        prediction_model = self.create_model(1)
        prediction_model.set_weights(self.model.get_weights())

        batch = np.zeros((1, self.time_len, 128), dtype="float")

        notes = deque(start_notes, self.time_len)
        
        prediction_model.reset_states()

        while True:
            batch[0, :, :] = notes
            prediction = prediction_model.predict_on_batch(batch)[0]
            # print(prediction)
            bool_predict = prediction > self.cutoff
            notes.append(bool_predict.astype("float"))
            yield bool_predict

def batch_generator(songs, time_len, batch_len):
    for (_, notes) in songs:
        notes = notes.copy()
        while (len(notes) - time_len) % batch_len != 0:
            notes.append(np.zeros(128))
        
        midi_input = np.asarray([notes[i:i + time_len] for i in range(0, len(notes) - time_len)])
        midi_output = np.asarray([notes[i + time_len] for i in range(0, len(notes) - time_len)])
        
        batches = []
        batch_count = int(len(midi_input) / batch_len)
        
        if batch_count == 0:
            continue

        for b in range(0, batch_count):
            b_start = b * batch_len
            b_end = b_start + batch_len
            batches.append((midi_input[b_start : b_end], midi_output[b_start : b_end]))
        
        yield batches