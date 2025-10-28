import os
import torch
from matplotlib import pyplot as plt

from dataset_loader import load_mutual_info_dataset

def plot_generated_images(X, num_corr_images=2, sample_indices=[0,1]):
    """
    Plots generated images from the dataset.
    Each row corresponds to a sample index, and each column corresponds to a correlated image modality.
    :param X: Numpy array of shape (num_samples, num_corr_images, height, width, channels)
    :param num_corr_images: Number of correlated images/modalities.
    :param sample_indices: List of sample indices to plot.
    :return: None
    """
    num_rows = len(sample_indices)
    plt.figure(figsize=(10, 2.2 * num_rows))

    for row_idx, sample_idx in enumerate(sample_indices):
        for j in range(num_corr_images):
            # compute subplot position: row 0→slots 1–2, row 1→slots 3–4
            subplot_index = row_idx * num_corr_images + j + 1
            ax = plt.subplot(num_rows, num_corr_images, subplot_index)
            image = X[sample_idx, j].transpose(1, 2, 0)
            ax.imshow(image)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f"Index {sample_idx}", fontsize=12, rotation=0, labelpad=40, va='center')
            # Optionally, add title above each image (uncomment below)
            ax.set_title(f"Modality {j}")

    plt.tight_layout()
    plt.show()



def plot_2x8_grid_modalities(dataset_dir: str, indices: list, output_path: str = "modalities_grid",
                             row_labels: tuple = ("Modality 1", "Modality 2")):
    """
    Plots and saves a 2x8 grid of images from the dataset,
    using both modalities for each given index.
    Row 1: modality 1, Row 2: modality 2, columns = indices.
    """
    assert len(indices) == 8, "You must supply exactly 8 indices."
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    X, *_ = load_mutual_info_dataset(dataset_dir)
    fig, axes = plt.subplots(2, 8, figsize=(20, 5), dpi=300)
    for col, idx in enumerate(indices):
        for row in range(2):
            img = X[idx, row].transpose(1,2,0)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=18, labelpad=35, rotation=0, va='center', ha='right')

    plt.tight_layout()
    pdf_path = output_path + ".pdf"
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0, dpi=600)
    print(f"Saved grid to {pdf_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Load the .npz file
    # dataset_dir = "../output_dir/datasets/CaseConstructive/"
    dataset_dir = "../output_dir/datasets/galaxy_images_curve_DAG/"


    X, Y, Noise, Sigma, Mu, labels = load_mutual_info_dataset(dataset_dir)

    # Example for PyTorch Loader
    x_train = torch.from_numpy(X)
    y_train = torch.from_numpy(Y)
    train_set = torch.utils.data.TensorDataset(x_train, y_train)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=4)

    num_corr_images = 2

    # specify exactly which dataset indices to plot
    sample_indices = [0, 1, 2, 3]

    # Plot the generated images
    plot_generated_images(X, num_corr_images=num_corr_images, sample_indices=sample_indices)


    plot_2x8_grid_modalities(
        dataset_dir=dataset_dir,
        indices=[1,2,3,4,5,6,7,9],
        output_path="figs/example_flow_outputs",
        row_labels=("Modality 1", "Modality 2")
    )
