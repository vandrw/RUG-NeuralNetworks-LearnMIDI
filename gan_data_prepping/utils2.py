import os
import sys
import shutil
from midi2img import *
    
folder_path = "/data/midi"
midi_extensions = ["mid"]

if not (os.path.exists(folder_path)):
    print('"', folder_path, '" does not exist!')
    sys.exit()

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

# [INFO] Converting  JJames-Last-Medley-Nr-5-(Medley).mid