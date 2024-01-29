"""Get the results from the nested sampling runs and combine them into a single file.

Get the dimension number and seed number from the file name
"results_ns_d{dim}_seed{seed}.txt"

Each file will have the following columns:
logZ, H, iterations, time


Put all this data into a single file with the following columns:
dim, seed, logZ, H, iterations, time
"""

import os
import re
import pandas as pd

# Directory containing the result files
results_directory = "results_ns"

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame(columns=["dim", "seed", "logZ", "H", "iterations", "time"])

# Regular expression pattern to match file names
pattern = re.compile(r"results_ns_d(\d+)_seed(\d+).txt")

# Iterate through the files in the directory
for filename in os.listdir(results_directory):
    # Match the filename pattern
    match = pattern.match(filename)
    if match:
        dim = int(match.group(1))
        seed = int(match.group(2))
        file_path = os.path.join(results_directory, filename)
        # Read data from the file into a DataFrame
        data = pd.read_csv(file_path, sep='\s+', header=None, names=["logZ", "H", "iterations", "time"])
        # Add dimension and seed columns
        data["dim"] = dim
        data["seed"] = seed
        # Reorder columns
        data = data[["dim", "seed", "logZ", "H", "iterations", "time"]]
        combined_data = pd.concat([combined_data, data], ignore_index=True)

# Save the combined data to an output file
combined_data.to_csv("combined_results.csv", index=False)
