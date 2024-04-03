from __future__ import annotations

import csv
import os
import random
import sys

target_dir = sys.argv[1]  #'/home/matt/work/data/lightning-test-2024'

# Get a list of all files in the target directory
files = os.listdir(f"{target_dir}/raw")

# Initialize the split percentages
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Shuffle the files randomly
random.shuffle(files)

# Calculate the number of files for each split
num_files = len(files)
num_train = int(num_files * train_split)
num_val = int(num_files * val_split)
num_test = num_files - num_train - num_val

# Create a list to store the split values
splits = ["train"] * num_train + ["val"] * num_val + ["test"] * num_test

# Create a list to store the identifier values
identifiers = [os.path.splitext(file)[0] for file in files]

# Combine the identifiers and splits into a list of tuples
data = list(zip(identifiers, splits))

# Write the data to the split.csv file
with open(f"{target_dir}/split.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["identifier", "split"])
    writer.writerows(data)
