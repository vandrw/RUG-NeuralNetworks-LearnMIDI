import os
import sys
import shutil
from midi2img import *

if len(sys.argv) == 1:
    print("Specify the location of the folder you want to convert midi files from...")
    print("Example: python main.py [path_to_folder]")
    sys.exit()
    
if len(sys.argv) > 2:
    print("More than one directoriy was specified. Aborting.")
    sys.exit()
    
folder_path = "/mnt/c/Docs/RUG/Second Year/2B/Neural Networks/RUG-NeuralNetworks-LearnMIDI/data/midi"
midi_extensions = ["mid"]

if not (os.path.exists(folder_path)):
    print('"', folder_path, '" does not exist!')
    sys.exit()

if len(sys.argv) == 2:
    print("The script will convert all the midi files in:")
    print(folder_path)
    ans = input("Do you wish to continue?[y/n] ")
    if not (ans == "y"):
        sys.exit()

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(folder_path):
    # print("Extracting from ", root.split(os.sep)[-1])
    for file_name in files:
        extension = file_name.split(".")[-1]
        if extension.lower() in midi_extensions:
            print("[INFO] Converting ", file_name)
            full_midi_path = root + os.sep + file_name
            try:
                midi2image(full_midi_path)
            except Exception as e:
                print("[FAIL] " + str(e))
                continue