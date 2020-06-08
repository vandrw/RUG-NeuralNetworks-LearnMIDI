import tensorflow.keras as tfk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

class MidiModel:
    def __init__(self):
        try:
            self.model = tfk.models.load_model("model/final/midi.h5")
            print("Model successfuly loaded.")
        except:
            print("The model could not be loaded! Creating a new one.")
            self.model = self.create_model()
        
        self.input_shape = (128,) # TODO Make this dynamic.
            
    def create_model(self):
        model = Sequential()

        model.add(LSTM(256, input_shape=(
            self.input_shape[0], self.input_shape[1]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        return model

    def train(self, midi_input, midi_output):
        filepath = "model/ckpts/weights-{epoch:02d}-{loss:.4f}.hdf5"

        # Checkpoints for creating a plot after training. 
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', 
            verbose=1,        
            save_best_only=True,        
            mode='min'
        )

        self.model.fit(midi_input, midi_output, epochs=200, batch_size=64, callbacks=[checkpoint])

        self.model.save("model/final/midi.h5")
    
    def test(self):
        # TODO
        pass
        
def string_to_bits(string):
    array = np.zeros(128, dtype="float")
    i = 0
    for char in string:
        bits = int(char, 16)
        for j in range(3, -1, -1):
            array[i] = (bits >> j) & 1
            i += 1
    return array

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8', header=None)
    data.columns = ["midi"]
    data = data[~data.midi.str.contains("#")]

    bit_data = []
    for row in data["midi"]:
        bit_data.append(string_to_bits(row))

    bit_data = np.asarray(bit_data, dtype=np.int32)
    
    # np.savetxt("data/out_all_128.txt", bit_data)
    return bit_data
    

if __name__ == "__main__":
    # midiModel = MidiModel()
    try:
        data = load_data("data/out-all.txt")
    except:
        print("Data could not be imported. Check 'data/out-all.txt'...")
    
    print(data[:5], data.shape)
