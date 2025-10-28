import numpy as np
import glob

def load_mutual_info_dataset(dataset_dir,):
    """
    Loads a dataset from .npz files generated using the mutual information framework.
    :param dataset_dir: Directory containing the .npz files.
    :return: Tuple of numpy arrays: (X, Y, Noise, Sigma, Mu, labels)
    """
    npzfiles = glob.glob(f"{dataset_dir}/*.npz")
    npzfiles.sort()

    if not npzfiles:
        raise RuntimeError("No files found!")

    print_freq_iterations = 2

    X = []
    Y = []
    Noise = []
    Sigma = []
    Mu = []
    labels = []

    for i, npzfile in enumerate(npzfiles):
        if i != 0 and (i + 1) % (len(list(npzfiles)) // print_freq_iterations) == 0:
            print(f"Progress: [Step [{i + 1}/{len(list(npzfiles))}]")
            print("------------------------------", flush=True)
        X.append(np.load(npzfile)['X'])
        Y.append(np.load(npzfile)['Y'])
        Noise.append(np.load(npzfile)['Noise'])
        Sigma.append(np.load(npzfile)['Sigma'])
        Mu.append(np.load(npzfile)['Mu'])
        labels.append(np.load(npzfile)['labels'])

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Noise = np.concatenate(Noise, axis=0)
    Sigma = np.concatenate(Sigma, axis=0)
    Mu = np.concatenate(Mu, axis=0)
    labels = np.concatenate(labels, axis=0)

    return X, Y, Noise, Sigma, Mu, labels