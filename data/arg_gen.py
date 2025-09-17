import os

name2id = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243"
}

id2name = {v: k for k, v in name2id.items()}

obj_ids2id = {0: "02691156", 1: "02958343", 2: "03001627", 3: "03636649", 4: "04099429"}

directory = f"/work/mech-ai/jrrade/Tri-plane"

for id in obj_ids2id:
    print(id, len(os.listdir(f"{directory}/{obj_ids2id[id]}")))

# for each obj_ids, create 20 args_<id>_i.txt files, where i = [0,19] and divide subdirectories into 20 files
for id in obj_ids2id:
    subdirectories = os.listdir(f"{directory}/{obj_ids2id[id]}")
    for i in range(20):
        with open(f"./args/args_{id}_{i}.txt", "w") as f:
            for subdir in subdirectories[i::20]:
                f.write(f"{directory}/{obj_ids2id[id]}/{subdir}\n")