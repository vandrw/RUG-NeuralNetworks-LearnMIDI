import tensorflow.keras as tfk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

class MidiModel:
    def __init__(self, time_len, batch_len):
        self.time_len = time_len        
        self.batch_len = batch_len            
        #try:
        #    self.model = tfk.models.load_model("model/final/midi.h5")
        #    print("Model successfuly loaded.")
        #except:
        #    print("The model could not be loaded! Creating a new one.")
        self.model = self.create_model()
        
        self.input_shape = (128,) # TODO Make this dynamic.
            
    def create_model(self):
        model = Sequential()

        model.add(Dense(64, batch_input_shape=(self.batch_len, self.time_len, 128), activation="relu"))
        model.add(Dropout(0.3))
        model.add(LSTM(64, stateful=True))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.build(input_shape=(self.batch_len, self.time_len, 128))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # model.compile(loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='sgd')

        model.summary()

        return model

    def train(self, epochs, path, max_line_nr):

        for i in range(0, epochs):
            print("Epoch: ", i + 1, "/", epochs)
            loss = 0
            batche_iter = note_batch_generator(path, self.time_len, self.batch_len)
            
            for (line_nr, batches) in batche_iter:
                if line_nr > max_line_nr:
                    break

                self.model.reset_states()
                for (x, y) in batches:
                    batch_loss = self.model.train_on_batch(x, y=y)
                    loss = loss * 0.9 + batch_loss * 0.1
                    print("nr: %5d, loss: %02.4f" % (line_nr, loss), end='\r')

    def save(self):
        self.model.save("model/final/midi.h5")
    
    def test(self):
        # TODO
        pass
        
def note_batch_generator(path, time_len, batch_len):
    input_file = open(path, "r")
    
    notes = []
    line_nr = 0
    for line in input_file:
        if line[0] == '#':
            while (len(notes) - time_len) % batch_len != 0:
                notes.append(np.zeros(128))
            midi_input = np.asarray([notes[i:i + time_len] for i in range(0, len(notes) - time_len)])
            midi_output = np.asarray([notes[i + time_len] for i in range(0, len(notes) - time_len)])
            batches = []
            batch_count = int(len(midi_input) / batch_len)
            for b in range(0, batch_count):
                batches.append((midi_input[b * batch_len: (b + 1) * batch_len], midi_output[b * batch_len: (b + 1) * batch_len]))
            yield (line_nr, batches)
            notes.clear()
        else:
            bits = string_to_bits(line)
            if not np.all(bits == 0.0):
                notes.append(string_to_bits(line))
        line_nr += 1

def string_to_bits(string):
    array = np.zeros(128, dtype="float")
    i = 0
    for char in string:
        if char == '\n':
            continue
        bits = int(char, 16)
        for j in range(3, -1, -1):
            array[i] = (bits >> j) & 1
            i += 1
    return array

if __name__ == "__main__":
    midiModel = MidiModel(32, 32)
    
    midiModel.train(3, "data/out-all.txt", max_line_nr = 1000000)

    midiModel.save()

    # notes = []
    # input_file = open("data/out-all.txt", "r")
    # for line in input_file.readlines()[:50000]:
    #     if line[0] != '#':
    #         bits = string_to_bits(line)
    #         if not np.all(bits == 0.0):
    #             notes.append(string_to_bits(line))
    
    # time_len = 32
    # batch_len = 32

    # midi_input = np.asarray([notes[i:i + time_len] for i in range(0, len(notes) - time_len)])
    # midi_output = np.asarray([notes[i + time_len] for i in range(0, len(notes) - time_len)])
    
    # batch_count = int(len(midi_input) / batch_len)
    # midi_input = midi_input[0:batch_count * batch_len,:] #.reshape(-1, batch_len, time_len, 128)
    # midi_output = midi_output[0:batch_count * batch_len] #.reshape(-1, batch_len, 128)

    # print("midi_input.shape: ", midi_input.shape)
    # print("midi_output.shape: ", midi_output.shape)

    # midiModel.train(midi_input, midi_output)
