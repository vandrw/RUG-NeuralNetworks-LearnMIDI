import numpy as np

def read_songs(path):
    input_file = open(path, "r")
    
    notes = []
    line_nr = 0
    for line in input_file:
        if line[0] == '#':
            if len(notes) > 0:
                yield (line_nr, notes)
                notes = []
        else:
            bits = string_to_bits(line)
            notes.append(bits)
        line_nr += 1
    
    if len(notes) > 0:
        yield (line_nr, notes)


def note_batch_generator(path, time_len, batch_len):
    for line_nr, notes in read_songs(path):
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
        
        yield (line_nr, batches)
        

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

def bits_to_string(bits):
    string = ""
    i = 0
    one_char = 0
    for bit in bits:
        one_char <<= 1
        if bit:
            one_char += 1
        i += 1

        if i == 4:
            string += "%X" % one_char
            i = 0
            one_char = 0

    return string