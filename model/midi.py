from midiutil import MIDIFile
from itertools import chain
import numpy as np

def create_midi_file(steps, bpm = 60):
    track    = 0
    channel  = 0
    time     = 0    # In beats
    volume   = 100  # 0-127, as per the MIDI standard

    midi_file = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                        # automatically)
    midi_file.addTempo(track, time, bpm)

    on_count = np.zeros(128, dtype=int)
    time = 0

    for step in chain(steps, [np.zeros(128, dtype=bool)]):
        on_count[step] += 1
        complete_notes = np.argwhere(np.logical_and(on_count != 0, np.logical_not(step)))
        for note in complete_notes:
            idx = note[0]
            note_dur = on_count[idx]
            on_count[idx] = 0
            midi_file.addNote(track, channel, idx, (time - note_dur) / 8.0, note_dur / 8.0, volume)
        
        time += 1
    
    return midi_file

def create_and_save_midi_file(steps, path, bpm = 60):
    midi_file = create_midi_file(steps, bpm)

    with open(path, "wb") as output_file:
        midi_file.writeFile(output_file)
    
    print("Saved midi file: %s" % path)

if __name__ == "__main__":
    steps = np.zeros((2, 128), dtype=bool)
    steps[:,60] = True
    steps[0,61] = True
    steps[1,62] = True

    create_and_save_midi_file(steps, "test.mid")