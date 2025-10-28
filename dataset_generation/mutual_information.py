import torch
import numpy as np


def create_mutual_info_covariance(num_corr_images=2, dim=3 * 32 * 32, cov_images=None, cov_theta=None, theta_var=None,
                                   epsilon=0, delta=1e-6, device=None):
    """
    This function creates a covariance matrix for a multivariate Gaussian distribution with specific properties.
    :param device: Device to create the covariance matrix on (e.g., 'cuda' or 'cpu').
    :param num_corr_images: Number of correlated images to generate.
    :param dim: Dimension of each image (e.g., 3 * 32 * 32 for CIFAR-10).
    :param cov_images: 2D array-like of shape (num_corr_images, num_corr_images) specifying the covariance between each pair of
                       images. Diagonal elements are the variances for each image and off-diagonals are the covariances.
    :param cov_theta: List or scalar specifying the cross-covariance between each image and theta.
                      If scalar, all images will have the same covariance with theta.
    :param theta_var: Scalar, variance for theta.
    :param epsilon: Small value added to the diagonal for numerical stability.
    :return: mu (mean vector) and Sigma (covariance matrix).
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if cov_images is None:
        Sigma = create_theta_mi_covariance(num_corr_images=num_corr_images, dim=dim, theta_var=theta_var,
                                           cov_theta=cov_theta, delta=delta, device=device)
    else:
        # Process cov_theta.
        # Initialize sigma_itheta list
        sigma_itheta_list = []

        if cov_theta is None:
            # Randomly generate each sigma_itheta vector
            for _ in range(num_corr_images):
                sigma_itheta = torch.randn(dim, device=device)
                sigma_itheta_list.append(sigma_itheta)
        elif np.isscalar(cov_theta):
            # Use the same scalar value to construct identical vectors
            for _ in range(num_corr_images):
                sigma_itheta = torch.full((dim,), cov_theta, device=device)
                sigma_itheta_list.append(sigma_itheta)
        elif isinstance(cov_theta, (list, tuple)) and all(np.isscalar(val) for val in cov_theta) and len(
                cov_theta) == num_corr_images:
            # Each entry corresponds to a scalar for each image
            for val in cov_theta:
                sigma_itheta = torch.full((dim,), val, device=device)
                sigma_itheta_list.append(sigma_itheta)
        else:
            raise ValueError(
                "cov_theta must be None, a scalar, or a list of scalars with length equal to num_corr_images.")

        # Process cov_images.
        if cov_images is None:
            # Default: images are independent with unit variance.
            cov_images = np.eye(num_corr_images)
        else:
            cov_images = np.array(cov_images)
            if cov_images.shape != (num_corr_images, num_corr_images):
                raise ValueError("cov_images must have shape (num_corr_images, num_corr_images).")

        # Build the image covariance block.
        image_blocks = []
        for i in range(num_corr_images):
            row_blocks = []
            for j in range(num_corr_images):
                # Each block is (cov_images[i,j]) * I_d.
                row_blocks.append(cov_images[i, j] * torch.eye(dim, device=device))
            row = torch.cat(row_blocks, dim=1)
            image_blocks.append(row)
        Sigma_images = torch.cat(image_blocks, dim=0)

        # Theta's block: 1x1 matrix.
        if theta_var is None:
            theta_var = 1.0
        Theta_block = torch.tensor([[theta_var]], device=device)

        # Form the full covariance matrix.
        Sigma = torch.block_diag(Sigma_images, Theta_block)

        # Fill in the cross-covariance blocks between each image and theta.
        ones_image = torch.ones(dim, device=device)
        for i in range(num_corr_images):
            start = i * dim
            end = (i + 1) * dim
            sigma_i = sigma_itheta_list[i]
            Sigma[start:end, -1] = sigma_i
            Sigma[-1, start:end] = sigma_i


    if epsilon > 0:
        # Add a small epsilon to the diagonal for numerical stability.
        Sigma += epsilon * torch.eye(Sigma.shape[0], device=device)

    #### Temporary Addition to Manually Set a Block to a Constant Value ####
    # Sigma = modify_covariance_block(Sigma, block_i=0, block_j=1, constant=0.4, dim=dim, device=device)

    # Check if the covariance matrix is positive definite.
    if not is_positive_definite(Sigma, atol=epsilon):
        theta_var_final = Sigma[-1, -1].item()
        min_theta = min_required_theta_var(Sigma)
        if theta_var_final <= min_theta:
            raise ValueError(
                f"Covariance matrix is not positive definite: "
                f"Theta variance = {theta_var_final:.2f} ≤ min required = {min_theta:.2f}. "
                f"Violates Schur complement condition. Try increasing theta_var."
            )
        else:
            raise ValueError("Covariance matrix is not positive definite. Issue is not with the Schur complement condition.")

    # Create mean vector.
    mu = torch.zeros(Sigma.shape[0], device=device)
    return mu, Sigma


def create_theta_mi_covariance(num_corr_images=2, dim=3 * 32 * 32, theta_var=None, cov_theta=None, delta: float = 1e-6, device='cpu'):
    """
    Given two dim-dimensional vectors sigma_1theta and sigma_2theta, construct a positive-definite covariance matrix Σ
    of size (2d+1)x(2d+1) such that Σ_{12} = sigma_1theta sigma_2theta^T / Σ_theta and Σ_theta > max(||sigma_1theta||^2, ||sigma_2theta||^2).

    :param num_corr_images: Number of correlated images
    :param dim: Dimension of each image (e.g., 3 * 32 * 32 for CIFAR-10)
    :param theta_var: Variance for theta. If None, it will be set to max(||sigma_1theta||^2, ||sigma_2theta||^2) + delta.
    :param cov_theta: List specifying the cross-covariance between each image and theta. Leave None to generate.
    :param delta: Small value added to the diagonal for numerical stability.
    :param device: Device to create the covariance matrix on (e.g., 'cuda' or 'cpu').
    :return: covariance matrix of shape (2d+1, 2d+1)
    """

    if num_corr_images != 2:
        raise ValueError(
            "Autogeneration of the covariance matrix only supports 2 modalities at this time. Please supply your own covariance values in this case.")

    if cov_theta:
        sigma_1theta = torch.full((dim,), cov_theta[0], device=device)
        sigma_2theta = torch.full((dim,), cov_theta[1], device=device)
    else:
        sigma_1theta = torch.randn(dim, device=device)
        sigma_2theta = torch.randn(dim, device=device)

    N_u = torch.linalg.norm(sigma_1theta).item() ** 2  # Squared Euclidean Norm of sigma_1theta
    N_v = torch.linalg.norm(sigma_2theta).item() ** 2

    max_of_N = max(N_u, N_v)

    if theta_var is None:
        Sigma_theta = max_of_N + delta  # To ensure Sigma_theta > max(||sigma_1theta||^2, ||sigma_2theta||^2)
    else:
        if theta_var > max_of_N:
            Sigma_theta = theta_var
        else:
            raise ValueError(f"theta_var must be greater than max(||sigma_1theta||^2, ||sigma_2theta||^2) (in this case: {max_of_N}).")

    # Compute the cross-block
    Sigma_12 = torch.outer(sigma_1theta, sigma_2theta) / Sigma_theta

    # Assemble the full covariance matrix
    size = 2 * dim + 1
    Sigma = torch.zeros(size, size, device=device)
    # Top-left block I_d
    Sigma[:dim, :dim] = torch.eye(dim)
    # Top-middle block Σ_{12}
    Sigma[:dim, dim:2 * dim] = Sigma_12
    # Top-right block sigma_1theta
    Sigma[:dim, -1] = sigma_1theta
    # Middle-left block Σ_{21}
    Sigma[dim:2 * dim, :dim] = Sigma_12.T
    # Middle block I_d
    Sigma[dim:2 * dim, dim:2 * dim] = torch.eye(dim)
    # Middle-right block sigma_2theta
    Sigma[dim:2 * dim, -1] = sigma_2theta
    # Bottom-left blocks sigma_1theta^T, sigma_2theta^T
    Sigma[-1, :dim] = sigma_1theta
    Sigma[-1, dim:2 * dim] = sigma_2theta
    # Bottom-right scalar Σ_theta
    Sigma[-1, -1] = Sigma_theta

    if not is_positive_definite(Sigma):
        raise ValueError("Covariance matrix is not positive definite. Try increasing the delta value.")

    return Sigma


def modify_covariance_block(Sigma, block_i, block_j, constant, dim, device):
    """
    Modifies the block of the covariance matrix Sigma corresponding to the images
    with indices block_i and block_j to be a constant matrix (all elements equal to constant).
    Also sets the symmetric block (block_j, block_i) to the same value.

    Parameters:
      Sigma: The covariance matrix (a torch tensor).
      block_i: The index of the first image block (0-indexed).
      block_j: The index of the second image block (0-indexed).
      constant: The constant value to fill the block.
      dim: The dimension of each image block.
      device: The torch device (e.g., 'cuda' or 'cpu').

    Returns:
      The modified covariance matrix.
    """
    # Create a block filled with the constant value.
    block = constant * torch.ones(dim, dim, device=device)

    # Set block (block_i, block_j)
    Sigma[block_i * dim:(block_i + 1) * dim, block_j * dim:(block_j + 1) * dim] = block
    # Set the symmetric block (block_j, block_i) if needed.
    if block_i != block_j:
        Sigma[block_j * dim:(block_j + 1) * dim, block_i * dim:(block_i + 1) * dim] = block

    return Sigma


def is_positive_definite(matrix: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """
    Returns True iff `matrix` is (numerically) symmetric and admits a Cholesky factorization with no non-positive pivots.
    Behavior mirrors torch.distributions.constraints.positive_definite:
      1) symmetry within (atol, rtol)
      2) torch.linalg.cholesky_ex → info == 0
    :param matrix: a square (or batch of square) Tensor.
    :param atol: tolerances for the symmetry check, same value as PyTorch uses.
    :param rtol: tolerances for the symmetry check.
    :return: boolean
    """

    # Symmetry check
    if not torch.allclose(matrix, matrix.mT, atol=atol,rtol=rtol):
        return False

    # Cholesky attempt
    # torch.linalg.cholesky_ex returns (L, info), where info==0 means success.
    # if batched, info is a tensor of ints; we require *all* to be zero
    return torch.linalg.cholesky_ex(matrix).info.eq(0).item()

def min_required_theta_var(Sigma: torch.Tensor) -> float:
    """
    Given a full covariance matrix Sigma (with theta as the last variable), compute the minimum required theta variance
    (sigma_theta^2) such that the matrix remains positive definite. Essentially ensuring Schur's complement is positive.

    :param Sigma: Covariance matrix of shape (N+1, N+1)
    :return: Minimum required variance for theta
    """
    # Split Sigma into A (top-left), B (top-right), theta_var (bottom-right)
    A = Sigma[:-1, :-1]  # Covariance of the non-theta variables
    B = Sigma[:-1, -1]   # Cross-covariance between non-theta variables and theta

    A_inv = torch.linalg.inv(A)
    min_theta_var = torch.dot(B, A_inv @ B).item()
    return min_theta_var
