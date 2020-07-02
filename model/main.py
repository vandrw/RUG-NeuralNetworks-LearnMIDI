from itertools import islice
import numpy as np

from model import MidiModel
from input import note_batch_generator, bits_to_string, read_songs
from midi import create_and_save_midi_file

if __name__ == "__main__":
    data_path = "data/out-off12.txt"
    time_len = 16
    batch_len = 64

    midiModel = MidiModel(time_len, batch_len, "model/final/midi-16-64-off12-adagrad-4.h5")
    for (line_nr, song) in read_songs(data_path):
        notes = []
        for bits in islice(midiModel.predict(song[:time_len]), 8*40):
            # print(bits_to_string(bits))
            # print (bits)
            notes.append(bits)
        create_and_save_midi_file(notes, "model/songs/line_%d.mid" % line_nr)

    # midiModel = MidiModel(time_len, batch_len)
    # midiModel.train(15, data_path, max_line_nr = 100000)
    # midiModel.save("model/final/midi-16-64-off12-adagrad-5.h5")

    # notes = []
    # input_file = open("data/out-all.txt", "r")
    # for line in input_file.readlines()[:50000]:
    #     if line[0] != '#':
    #         bits = string_to_bits(line)
    #         if not np.all(bits == 0.0):
    #             notes.append(string_to_bits(line))
    

    # midi_input = np.asarray([notes[i:i + time_len] for i in range(0, len(notes) - time_len)])
    # midi_output = np.asarray([notes[i + time_len] for i in range(0, len(notes) - time_len)])
    
    # batch_count = int(len(midi_input) / batch_len)
    # midi_input = midi_input[0:batch_count * batch_len,:] #.reshape(-1, batch_len, time_len, 128)
    # midi_output = midi_output[0:batch_count * batch_len] #.reshape(-1, batch_len, 128)

    # print("midi_input.shape: ", midi_input.shape)
    # print("midi_output.shape: ", midi_output.shape)

    # midiModel.train(midi_input, midi_output)

# Model3: Trained on all testing data generated using: `cargo run --release -- -m
# 12 ~/data/datasets/midis/ ../data/out-off16-final2.txt`. 
# 
# Model:
# model.add(Dense(64, 
#                 batch_input_shape=(self.batch_len, self.time_len, 128), 
#                 activation="relu", 
#                 kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.3))
# model.add(LSTM(80, 
#                 stateful=True,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 recurrent_regularizer=regularizers.l2(0.01), 
#                 bias_regularizer=None,
#                 activity_regularizer=regularizers.l2(0.01),))
# model.add(Dropout(0.3))
# model.add(Dense(128, 
#                 activation="relu", 
#                 kernel_regularizer=regularizers.l2(0.02)))
# 
# model.build(input_shape=(self.batch_len, self.time_len, 128))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# 
# batch, time = 32, 32