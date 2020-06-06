import tensorflow.keras as tfk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
# import numpy as np

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
        
if __name__ == "__main__":
    midiModel = MidiModel()