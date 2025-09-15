import os

directory = "/work/mech-ai/jrrade/Tri-plane/02691156"
subdirectories = os.listdir(directory)

# create 20 args.txt files and divide subdirectories into 20 files
for i in range(20):
    with open(f"args_{i}.txt", "w") as f:
        for subdir in subdirectories[i::20]:
            f.write(subdir + "\n")