import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def true_lnz(dim, v=0.1):
    return (dim / 2) * (np.log(v) - np.log(1 + v))



# Load the combined data from the CSV file
combined_data = pd.read_csv("combined_results.csv")

# Create three histograms for logZ, one for each dimension
plt.figure(figsize=(12, 4))
for i, dim in enumerate([1,20,100]):
    plt.subplot(1, 3, i+1)

    dim_data = combined_data[combined_data['dim'] == dim]
    plt.hist(dim_data['logZ'], bins=20, edgecolor='k')
    plt.title(f'Dimension {dim}')
    plt.xlabel('logZ')
    plt.ylabel('Frequency')
    # plot vertical line at true lnZ
    tru = true_lnz(dim)
    plt.axvline(tru, color='r', linestyle='--')


plt.tight_layout()
plt.show()
