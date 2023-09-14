import glob

import numpy as np

OUTDIR = "out"


def get_data_from_file(f):
    """Get data from file"""
    data = []
    with open(f, "r") as f:
        for line in f:
            char0 = line[0]
            if char0.isdigit() or char0 == "-":
                data.append([float(x) for x in line.split()])
    if len(data) > 0:
        return np.array(data)


def get_combined_data(files):
    """Combine files into a single file"""
    data = []
    for f in files:
        di = get_data_from_file(f)
        if di is not None:
            data.append(di)
    if len(data) > 0:
        return np.vstack(data)


def main():
    data = []
    for d in [1, 20, 100]:
        data_di = get_combined_data(glob.glob(f"{OUTDIR}/*_d{d}_v*.dat"))
        if data_di is not None:
            data_di = np.hstack([np.ones((data_di.shape[0], 1)) * d, data_di])
            data.append(data_di)
    if len(data) > 0:
        data = np.vstack(data)
        # counts of samples for each dimension
        counts = np.unique(data[:, 0], return_counts=True)
        print(f"Saving combined data")
        np.savetxt(
            f"{OUTDIR}/nested_sampling_lnzs.dat",
            data,
            header="dim ns_lnz ns_lnz_err",
            fmt=["%03d", "%.5e", "%.5e"],
        )
        for d, c in zip(*counts):
            print(f"Dimension {int(d)}: {c} samples")


if __name__ == "__main__":
    main()
