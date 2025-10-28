import os
from glob import glob
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def run_coefficient_validation(N_u_theta, N_u, N_mod, mod_dim, eta, rho_theta, rho, T_vectors, DAG_theta, DAG_ul):
    """
    Validate the sizes of the coefficient tensors.
    """

    if eta.shape[0] != N_u_theta:
        raise ValueError(f"Size of eta ({eta.shape[0]}) does not match number of theta proto-latents ({N_u_theta}).")
    if rho_theta.shape[0] != N_u_theta:
        raise ValueError(f"Size of rho_theta ({rho_theta.shape[0]}) does not match number of theta proto-latents ({N_u_theta}).")
    if rho.shape[0] != N_u:
        raise ValueError(f"Size of rho ({rho.shape[0]}) does not match number of modalities ({N_mod}).")
    if T_vectors.shape != (N_u_theta, mod_dim, N_mod):
        raise ValueError(f"Size of T_vectors {T_vectors.shape} does not match expected shape ({N_u_theta}, {mod_dim}, {N_mod}).")
    if DAG_theta.shape != (N_mod, N_u_theta):
        raise ValueError(f"Size of DAG_theta {DAG_theta.shape} does not match expected shape ({N_mod}, {N_u_theta}).")
    if DAG_ul.shape != (N_mod, N_u):
        raise ValueError(f"Size of DAG_ul {DAG_ul.shape} does not match expected shape ({N_mod}, {N_u}).")


def generate_rho_eta(N_u_theta, N_mod, eta_arg, rho_theta_arg, rho_arg, device=None):
    """
    Generate coefficients for the A matrix based on the model parameters.
    :return: eta, rho_theta, rho
    """
    # Example coefficients
    if isinstance(eta_arg, str):
        print(f"Eta Argument: {eta_arg}")
        # Handle different cases
        eta = torch.ones(N_u_theta, device=device)
    else:
        eta = torch.Tensor(eta_arg).to(device)

    if isinstance(rho_theta_arg, str):
        print(f"Rho Theta Argument: {rho_theta_arg}")
        # Handle different cases
        rho_theta = torch.ones(N_u_theta, device=device)
    else:
        rho_theta = torch.Tensor(rho_theta_arg).to(device)

    if isinstance(rho_arg, str):
        print(f"Rho Argument: {rho_arg}")
        # Handle different cases
        rho = torch.ones(N_mod, device=device)
    else:
        rho = torch.Tensor(rho_arg).to(device)

    return eta, rho_theta, rho


def load_images_for_T_vectors(directory, N_u_theta, N_mod, mod_dim, device=None):
    """
    Load images from a directory to form the T_vectors tensor of shape (N_u_theta, mod_dim, N_mod). Images must be 32x32
    RGB (or whatever matches mod_dim). Images are sorted by filename (for reproducibility).

    :param directory: String: Directory containing images.
    :param N_u_theta: Number of theta proto-latents.
    :param N_mod: Number of modalities.
    :param mod_dim: Dimensionality of each modality (should match image size, e.g. 3072 for 32x32x3).
    :param device: PyTorch device ('cpu' or 'cuda').
    :return: T_vectors (torch.Tensor): Landscape vectors of shape (N_u_theta, mod_dim, N_mod).
    """
    # Get all image file paths (jpg, png, jpeg)
    files = sorted(glob(os.path.join(directory, "*.png")) +
                   glob(os.path.join(directory, "*.jpg")) +
                   glob(os.path.join(directory, "*.jpeg")))
    expected_num = N_u_theta * N_mod
    if len(files) != expected_num:
        raise ValueError(
            f"Expected {expected_num} images in {directory}, but found {len(files)}."
        )
    # Load and flatten each image
    imgs = []
    for path in files:
        img = Image.open(path).convert("RGB")  # force RGB
        arr = np.asarray(img)
        if arr.shape != (32,32,3):
            raise ValueError(f"Image {path} must have shape (32,32,3), got {arr.shape}")
        arr = arr.transpose(2,0,1).reshape(-1)  # (3,32,32) -> (3072,)
        arr = arr.astype(np.float32) / 255.0 # Normalize to [0,1]
        imgs.append(arr)
    imgs = np.stack(imgs, axis=0)  # (N_u_theta*N_mod, 3072)
    # Reshape and swap axes: (N_u_theta, N_mod, mod_dim)
    imgs = imgs.reshape(N_u_theta, N_mod, mod_dim)
    # Final required shape: (N_u_theta, mod_dim, N_mod)
    imgs = np.transpose(imgs, (0,2,1))
    # Convert to torch
    T_vectors = torch.tensor(imgs, device=device, dtype=torch.float32)
    return T_vectors


def generate_T_vectors(N_u_theta, mod_dim, N_mod, T_vectors_method, device=None):

    """
    Generate landscape vectors L based on the specified method.

    :param N_u_theta: Number of theta proto-latents.
    :param mod_dim: Dimensionality of each modality.
    :param N_mod: Number of modalities.
    :param T_vectors_method: Method to generate L vectors. E.g. "ones", "random", or a path to a .npy file.
    :param device: PyTorch device ('cpu' or 'cuda').
    :return: T_vectors (torch.Tensor): Landscape vectors of shape (N_u_theta, mod_dim, N_mod).
    """

    if T_vectors_method == "ones" or None:
        T_vectors = torch.ones(N_u_theta, mod_dim, N_mod, device=device)
    elif T_vectors_method == "random":
        T_vectors = torch.randn(N_u_theta, mod_dim, N_mod, device=device)
    elif T_vectors_method == "debug":
        base = torch.arange(1, mod_dim + 1, dtype=torch.float32, device=device)  # shape: (mod_dim,)
        T_vectors = base.view(1, mod_dim, 1).expand(N_u_theta, mod_dim, N_mod).clone()  # shape: (N_u_theta, d, N_mod)
    elif T_vectors_method.endswith(".npy"):
        # Path to .npy file containing the T_vectors
        L_vectors_np = np.load(T_vectors_method)
        T_vectors = torch.tensor(L_vectors_np, device=device, dtype=torch.float32)
    else: # os.path.isdir(T_vectors_method):
        # New: Load images from directory
        image_tensor = load_images_for_T_vectors(
            directory=T_vectors_method,
            N_u_theta=N_u_theta,
            N_mod=N_mod,
            mod_dim=mod_dim,
            device=device
        )
        T_vectors = image_tensor
    # else:
    #     raise ValueError(f"Unknown T_vectors_method: {T_vectors_method}")

    return T_vectors


def build_A_matrix(N_u_theta, N_mod, N_u, mod_dim,
                   alpha, beta, eta, rho_theta, rho,
                   T_vectors=None, DAG_theta=None, DAG_ul=None, device=None):
    """
    Build the full A matrix.

    :param N_u_theta: Number of theta proto-latents.
    :param N_mod: Number of modalities.
    :param mod_dim: Dimensionality of each modality.
    :param alpha: Decay factor for modality proto-latents.
    :param beta: Decay factor for theta proto-latents.
    :param eta: Weight tensor for theta proto-latents w.r.t. u_theta (1D tensor of shape [N_u_theta]).
    :param rho_theta: Weight tensor for theta proto-latents (1D tensor of shape [N_u_theta]).
    :param rho: Weight tensor for modality proto-latents (1D tensor of shape [N_mod]).
    :param T_vectors: Landscape vectors (3D tensor of shape [N_u_theta, mod_dim, N_mod]). If None, defaults to ones.
    :param device: PyTorch device ('cpu' or 'cuda'). None uses 'cpu'.
    :return: A (torch.Tensor): Full A matrix of shape (1 + N_mod * mod_dim, N_u_theta + N_mod * mod_dim).
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if T_vectors is None:
        T_vectors = torch.ones(N_u_theta, mod_dim, N_mod, device=device)

    if DAG_theta is None:
        DAG_theta = torch.ones(N_mod, N_u_theta, device=device)
    else:
        DAG_theta = torch.tensor(DAG_theta, device=device)

    if DAG_ul is None:
        DAG_ul = torch.ones(N_mod, N_u, device=device)
    else:
        DAG_ul = torch.tensor(DAG_ul, device=device)

    # Checks to ensure inputs are valid
    run_coefficient_validation(N_u_theta, N_u, N_mod, mod_dim, eta, rho_theta, rho, T_vectors, DAG_theta, DAG_ul)

    # Top row: scalar l_theta = eta @ u_theta, zeros elsewhere
    top_row = torch.cat([eta, torch.zeros(N_u * mod_dim, device=device)], dim=0).unsqueeze(0)

    # Build remaining blocks
    l_rows = []
    for i in range(N_mod):
        # Theta proto-latents block
        # Todo: 0.0^0.0 should evaluate to 1. Make sure that if same modality, the pre-factor becomes 1.
        theta_block = torch.cat([
            DAG_theta[i, k] * rho_theta[k] * (beta ** -abs(i)) * T_vectors[k, :, i].view(mod_dim, 1)
            for k in range(N_u_theta)
        ], dim=1)

        # Modality proto-latents block
        latent_block = torch.cat([
            DAG_ul[i, j] * rho[j] * (alpha ** -abs(i - j)) * torch.eye(mod_dim, device=device)
            for j in range(N_u)
        ], dim=1)

        row_block = torch.cat([theta_block, latent_block], dim=1)
        l_rows.append(row_block)

    A = torch.cat([top_row] + l_rows, dim=0)
    return A


def calculate_cov_of_l(A: torch.Tensor) -> torch.Tensor:
    """
    Calculate the full covariance Cov(l) = A A^T by explicitly forming the product.
    :param A:  Matrix
    :return: Covariance matrix
    """
    return A @ A.T


def generate_row_orthonormal_matrix(m, n, device='cpu', dtype=torch.float32):
    """
    Generate a random (m x n) matrix with orthonormal rows using QR decomposition, such that A A^T = Identity.
    :param m: Number of rows (m <= n).
    :param n: Number of columns (m <= n).
    :param device: PyTorch device ('cpu' or 'cuda'). Default is 'cpu'.
    :param dtype: Data type for the matrix. Default is torch.float32.
    :return: A: A (m x n) matrix with orthonormal rows.
    """
    assert m <= n, "m must be less than or equal to n to have orthonormal rows."

    # Generate a random (n x m) matrix
    random_matrix = torch.randn(n, m, device=device, dtype=dtype)

    # Orthonormalize the columns via QR decomposition
    Q, _ = torch.linalg.qr(random_matrix, mode='reduced')  # Q: (n x m)

    # Transpose to get an (m x n) matrix with orthonormal rows
    A = Q.T  # shape (m x n)
    return A


def plot_gaussian_radii(X_list, labels=None, percentile_to_align=95.0, bins=50):
    """
    Plot the L2 norm (radius) of noise/latent vectors for multiple datasets.

    :param X_list: list of np.ndarray, each of shape (n_samples, d)
    :param labels: list of str, optional labels for each dataset
    :param percentile_to_align: float, the percentile to align the unit Gaussian distribution
    :param bins: int, number of histogram bins
    :return: None. Simply Plots.
    """

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(X_list))]

    d = X_list[0].shape[1] # Dimensionality
    theoretical_radius = np.sqrt(d) # Theoretical radius for unit  Gaussian distribution in d dimensions

    plt.figure(figsize=(10, 6))

    # Calculate the radii for the first dataset to determine bin edges
    max_radius = max(np.max(np.linalg.norm(X_list[0], axis=1)), 1e-6)  # avoid empty bin range
    # bin_edges = np.linspace(np.min(np.linalg.norm(X_list[0], axis=1)), max_radius, bins + 1)
    bin_edges = np.linspace(0, max_radius + 20, bins + 1)

    # Plot histograms for each dataset
    for X, label in zip(X_list, labels):
        radii = np.linalg.norm(X, axis=1) # L2 norm (Euclidean distance) for each sample
        plt.hist(radii, bins=bin_edges, alpha=0.5, density=True, label=label) # Density allows comparison of distributions even if sample sizes differ
        if percentile_to_align is not None:
            percentile_value = np.percentile(radii, percentile_to_align * 100)
            plt.axvline(percentile_value, linestyle='-.', label=f'{label} ({percentile_to_align * 100} Percentile) ≈ {percentile_value:.2f}')

    # Plot the theoretical radius for unit Gaussian
    plt.axvline(theoretical_radius, color='r', linestyle='--', label=f'Theoretical (For Unit), √d ≈ {theoretical_radius:.2f}')

    # plt.xlim(xmin=0)
    plt.title("Unit Gaussian vs. Constructed Gaussian Radii")
    plt.xlabel("Radius (L2 norm)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    print("Done!")
