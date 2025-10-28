import numpy as np


def safe_logdet(matrix):
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Matrix is not positive definite or has zero determinant.")
    return logdet


def reorder_constructive_to_sampling_sigma(Sigma, mod_dim, modalities, no_of_thetas):
    """
    Reorders a covariance matrix Sigma from constructive ordering to sampling ordering.

    Constructive ordering has theta variables first, followed by modality blocks, i.e.,
    [ Σ_θθ   Σ_θ1   Σ_θ2 ]
    [ Σ_1θ   Σ_11   Σ_12 ]
    [ Σ_2θ   Σ_21   Σ_22 ]

    Sampling ordering has modality blocks first, followed by theta variables, i.e.,
    [ Σ_11   Σ_12   Σ_1θ ]
    [ Σ_21   Σ_22   Σ_2θ ]
    [ Σ_θ1   Σ_θ2   Σ_θθ ]

    :param Sigma: NumPy array of shape (no_of_thetas + modalities * mod_dim, no_of_thetas + modalities * mod_dim)
    :param mod_dim: Dimension of each modality vector
    :param modalities: Number of modality/image blocks
    :param no_of_thetas: Number of theta variables
    :return: Reordered Sigma in sampling-style layout
    """
    if no_of_thetas != 1:
        raise NotImplementedError("Currently supports only 1 theta variable (no_of_thetas = 1).")

    total_dim = no_of_thetas + modalities * mod_dim

    # Indices for blocks in constructive ordering
    theta_idx = list(range(no_of_thetas))  # [0]
    mod_idx = list(range(no_of_thetas, total_dim))  # [1, 2, ..., total_dim-1]

    # Sampling ordering: modality blocks first, then theta
    new_order = mod_idx + theta_idx

    # Reorder rows and columns
    Sigma_reordered = Sigma[np.ix_(new_order, new_order)]
    return Sigma_reordered


def get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm="constructive"):
    """
    Extract blocks from a covariance matrix Sigma with multiple modalities and theta variables.

    :param Sigma: Covariance matrix of shape (modalities * mod_dim + no_of_thetas, modalities * mod_dim + no_of_thetas)
    :param mod_dim: Dimension of each modality vector
    :param modalities: Number of image/modalities blocks
    :param no_of_thetas: Number of theta variables
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: blocks: Dictionary of covariance blocks with keys like:
                sigma_11, sigma_12, sigma_13, sigma_32, sigma_33, etc.
                For e.g., for 2 modalities and 1 theta, sigma_33 should be a scalar.
    """

    blocks = {}
    total_blocks = modalities + no_of_thetas
    expected_dim = modalities * mod_dim + no_of_thetas
    if Sigma.shape != (expected_dim, expected_dim):
        raise ValueError(f"Expected Sigma shape ({expected_dim}, {expected_dim}), got {Sigma.shape}")

    if generation_paradigm == "sampling":
        for i in range(total_blocks):
            for j in range(total_blocks):
                # Determine slicing for i
                if i < modalities:
                    i_start, i_end = i * mod_dim, (i + 1) * mod_dim
                else:
                    i_start, i_end = modalities * mod_dim + (i - modalities), modalities * mod_dim + (
                                i - modalities) + 1

                # Determine slicing for j
                if j < modalities:
                    j_start, j_end = j * mod_dim, (j + 1) * mod_dim
                else:
                    j_start, j_end = modalities * mod_dim + (j - modalities), modalities * mod_dim + (
                                j - modalities) + 1

                key = f"sigma_{i + 1}{j + 1}"
                blocks[key] = Sigma[i_start:i_end, j_start:j_end]
    elif generation_paradigm == "constructive":
        for i in range(total_blocks):
            for j in range(total_blocks):
                # Adjusted order: theta blocks come first
                if i < no_of_thetas:
                    i_start, i_end = i, i + 1
                else:
                    i_start = no_of_thetas + (i - no_of_thetas) * mod_dim
                    i_end = i_start + mod_dim

                if j < no_of_thetas:
                    j_start, j_end = j, j + 1
                else:
                    j_start = no_of_thetas + (j - no_of_thetas) * mod_dim
                    j_end = j_start + mod_dim

                key = f"sigma_{i + 1}{j + 1}"
                blocks[key] = Sigma[i_start:i_end, j_start:j_end]

    return blocks


def get_gamma(Sigma, i, j, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Constructs the Gamma_{ij} block matrix from the covariance matrix Sigma.
    Gamma_{ij} = [[Σ_i,  Σ_ij],
                  [Σ_ij^T, Σ_j]]

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Block index i (1-based index, where modalities + 1 onward refers to theta blocks)
    :param j: Block index j (same indexing rule as i)
    :param mod_dim: Dimension of each modality block
    :param modalities: Number of modality/image blocks
    :param no_of_thetas: Number of theta scalar blocks
    :return: The Gamma matrix for blocks i and j
    """

    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)


    sigma_ii = blocks[f"sigma_{i}{i}"]
    sigma_jj = blocks[f"sigma_{j}{j}"]
    sigma_ij = blocks[f"sigma_{i}{j}"]
    sigma_ji = sigma_ij.T

    return np.block([
        [sigma_ii, sigma_ij],
        [sigma_ji, sigma_jj]
    ])


def compute_MI_scalars(N_u_theta, N_mod, N_u, alpha, beta, eta, rho_theta, rho, DAG_theta, DAG_ul):
    """
    Compute analytic scalars a, b, c, o, q, r_1, r_2, p, f to calculate the MI between theta, X_1, X_2 using the
    analytical equations.
    Note: Doesn't support Templates yet. Only works on bimodal case for now.

    :param N_u_theta: Number of theta proto-latents.
    :param N_mod: Number of modalities.
    :param N_u: Number of local-latents.
    :param alpha: Decay factor for modality proto-latents.
    :param beta: Decay factor for theta proto-latents.
    :param eta: Weight tensor for theta proto-latents w.r.t. u_theta (1D tensor of shape [N_u_theta]).
    :param rho_theta: Weight tensor for theta proto-latents (1D tensor of shape [N_u_theta]).
    :param rho: Weight tensor for modality proto-latents (1D tensor of shape [N_mod]).
    :param DAG_theta: DAG matrix for theta proto-latents (2D tensor of shape [N_mod, N_u_theta]).
    :param DAG_ul: DAG matrix for modality proto-latents (2D tensor of shape [N_mod, N_u]).
    :return: Dictionary of scalars.
    """
    # --- Theta term ---
    a = np.sum(eta ** 2)

    # --- Shared proto-latent terms ---
    # r_i for each modality
    r = []
    o = []
    b = []
    for i in range(N_mod):
        # r_i = sum_k eta_k * DAG_theta[i,k] * rho_theta[k] * beta^{-|i - k|}
        r_i = 0.0
        o_i = 0.0
        for k in range(N_u_theta):
            theta_weight = DAG_theta[i][k] * rho_theta[k] * (beta ** -abs(i))
            r_i += eta[k] * theta_weight
            o_i += theta_weight ** 2
        r.append(r_i)
        o.append(o_i)
        # local-latent part for b_i
        local_b = 0.0
        for j in range(N_u):
            local_b += (DAG_ul[i][j] * rho[j] * (alpha ** -abs(i - j))) ** 2
        b.append(o_i + local_b)

    # --- Cross terms (between modalities) ---
    # p = sum_k DAG_theta[0,k] * rho_theta[k] * ... * DAG_theta[1,k] ...
    p = 0.0
    for k in range(N_u_theta):
        theta1 = DAG_theta[0][k] * rho_theta[k] * (beta ** -abs(0))
        theta2 = DAG_theta[1][k] * rho_theta[k] * (beta ** -abs(1))
        p += theta1 * theta2

    # If local-latent cross-terms are present between i and j, add here; otherwise, f = p
    f = p
    # Add local-latent cross-terms to f
    for j in range(N_u):
        local1 = DAG_ul[0][j] * rho[j] * (alpha ** -abs(0 - j))
        local2 = DAG_ul[1][j] * rho[j] * (alpha ** -abs(1 - j))
        f += local1 * local2

    # --- q and c for second modality ---
    q = o[1]
    c = b[1]

    return dict(
        a=a,
        b=b[0],      # for first modality
        c=c,         # for second modality
        o=o[0],      # for first modality
        q=q,         # for second modality
        r_1=r[0],
        r_2=r[1],
        p=p,
        f=f
    )

def _compute_mutual_info_analytical(Sigma, i, j, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Analytically computes the conditional mutual information I(i; j) for Gaussian variables using closed-form formulas.
    Note: Does not work when --T_vectors_method flag is used because derived formulas assume all-one template vectors
    for sub-block symmetry.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks
    :param no_of_thetas: Number of scalar theta variables
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: Scalar value of conditional mutual information I(Xi; Xj)
    """
    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)

    # Block names are like 'sigma_12', etc.
    if i == 1 and j != 1:  # theta vs modality
        a = blocks["sigma_11"][0, 0]
        r = blocks[f"sigma_1{j}"][0, 0]
        Sigma_jj = blocks[f"sigma_{j}{j}"]
        b = np.diag(Sigma_jj)[0]
        o = Sigma_jj[0, 1] if mod_dim > 1 else 0.0
        inside = 1 - mod_dim * r**2 / (a * (b + (mod_dim-1)*o))
        if not (0 < inside < 1):
            raise ValueError(f"Argument to log is out of range: {inside}")
        return -0.5 * np.log(inside)

    elif j == 1 and i != 1:  # modality vs theta (symmetry)
        return _compute_mutual_info_analytical(Sigma, j, i, mod_dim, modalities, no_of_thetas, generation_paradigm)

    elif i != 1 and j != 1:  # modality vs modality
        Sigma_ii = blocks[f"sigma_{i}{i}"]
        b = np.diag(Sigma_ii)[0]
        o = Sigma_ii[0, 1] if mod_dim > 1 else 0.0
        B = b + (mod_dim-1)*o

        Sigma_jj = blocks[f"sigma_{j}{j}"]
        c = np.diag(Sigma_jj)[0]
        q = Sigma_jj[0, 1] if mod_dim > 1 else 0.0
        C = c + (mod_dim-1)*q

        Sigma_ij = blocks[f"sigma_{i}{j}"]
        f = np.diag(Sigma_ij)[0]
        p = Sigma_ij[0, 1] if mod_dim > 1 else 0.0

        num1 = (b - o) * (c - q)
        den1 = (b - o) * (c - q) - (f - p) ** 2
        term1 = 0.5 * (mod_dim - 1) * np.log(num1 / den1)

        num2 = B * C
        F = f + (mod_dim - 1) * p
        den2 = B * C - F ** 2
        term2 = 0.5 * np.log(num2 / den2)

        if den1 <= 0 or den2 <= 0 or num1 <= 0 or num2 <= 0:
            raise ValueError("Invalid arguments to log in MI(Xi;Xj)")
        return term1 + term2

    else:
        raise ValueError("Invalid block indices or not defined for (theta, theta).")


def _compute_mutual_info_numerical(Sigma, i, j, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Numerically computes the conditional mutual information I(i; j) for Gaussian variables using log-determinants.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks (default: 2)
    :param no_of_thetas: Number of scalar theta variables (default: 1)
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: Scalar value of conditional mutual information I(Xi; Xj)
    """

    # Compute necessary blocks
    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    Sigma_i = blocks[f"sigma_{i}{i}"]
    Sigma_j = blocks[f"sigma_{j}{j}"]
    Gamma_ij = get_gamma(Sigma, i, j, mod_dim, modalities, no_of_thetas, generation_paradigm)


    logdet_Sigma_i = safe_logdet(Sigma_i)
    logdet_Sigma_j = safe_logdet(Sigma_j)
    logdet_Gamma_ij = safe_logdet(Gamma_ij)

    # Mutual Information
    MI = 0.5 * (logdet_Sigma_i + logdet_Sigma_j - logdet_Gamma_ij)

    return MI


def compute_mutual_info(Sigma, i, j, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive", method="numerical"):
    """
    Computes the mutual information I(i; j) for Gaussian variables using either numerical or analytical methods.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks
    :param no_of_thetas: Number of scalar theta variables
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :param method: Method to compute mutual information, either "numerical" or "analytical"
    :return: Scalar value of mutual information I(Xi; Xj)
    """
    if method == "numerical":
        return _compute_mutual_info_numerical(Sigma, i, j, mod_dim, modalities, no_of_thetas, generation_paradigm)
    elif method == "analytical":
        return _compute_mutual_info_analytical(Sigma, i, j, mod_dim, modalities, no_of_thetas, generation_paradigm)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_X_theta_MI(Sigma, i, j, k, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Computes the conditional mutual information I((X_j, X_k); θ_i) for Gaussian variables using log-determinants.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks (default: 2)
    :param no_of_thetas: Number of scalar theta variables (default: 1)
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: Scalar value of conditional mutual information I(X1; X2 | θ)
    """

    # Compute necessary blocks
    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    Sigma_i = blocks[f"sigma_{i}{i}"]
    Gamma_jk = get_gamma(Sigma, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm) # 2nd and 3rd blocks are X2 and X3 respectively

    # Log-determinants

    logdet_Sigma = safe_logdet(Sigma)
    logdet_Sigma_i = safe_logdet(Sigma_i)
    logdet_Gamma_jk= safe_logdet(Gamma_jk)

    # Mutual Information
    MI = 0.5 * (logdet_Gamma_jk + logdet_Sigma_i - logdet_Sigma)


    return MI


def compute_Xj_theta_Xk_MI(Sigma, i, j, k, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Computes the conditional mutual information I(X_j ; θ_i | X_k) for Gaussian variables using log-determinants.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks (default: 2)
    :param no_of_thetas: Number of scalar theta variables (default: 1)
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: Scalar value of conditional mutual information I(X1; X2 | θ)
    """

    I_X_theta = compute_X_theta_MI(Sigma, i, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm)

    I_Xk_theta = compute_mutual_info(Sigma, k, i, mod_dim, modalities, no_of_thetas, generation_paradigm)

    MI = I_X_theta - I_Xk_theta

    # # Manual calculation
    # # Compute necessary blocks
    # blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # Gamma_jk = get_gamma(Sigma, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # Gamma_ki = get_gamma(Sigma, k, i, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # Sigma_k = blocks[f"sigma_{k}{k}"]
    #
    # # Log-determinants
    # logdet_Sigma_k = safe_logdet(Sigma_k)
    # logdet_Gamma_ki = safe_logdet(Gamma_ki)
    # logdet_Gamma_jk = safe_logdet(Gamma_jk)
    # logdet_Sigma = safe_logdet(Sigma)
    #
    # # Mutual Information
    # MI = -0.5 * (logdet_Gamma_jk + logdet_Gamma_ki - logdet_Sigma - logdet_Sigma_k)

    return MI


def compute_Xj_Xk_i_MI(Sigma, i, j, k, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Computes the conditional mutual information I(X_j ; X_k ; θ_i) for Gaussian variables using log-determinants.

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks (default: 2)
    :param no_of_thetas: Number of scalar theta variables (default: 1)
    :param generation_paradigm: Paradigm for block extraction, either "constructive" or "sampling"
    :return: Scalar value of conditional mutual information I(X1; X2 | θ)
    """


    I_Xj_theta = compute_mutual_info(Sigma, j, i, mod_dim, modalities, no_of_thetas, generation_paradigm)
    I_Xk_theta = compute_mutual_info(Sigma, k, i, mod_dim, modalities, no_of_thetas, generation_paradigm)
    I_X_theta = compute_X_theta_MI(Sigma, i, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm)

    MI = I_Xj_theta + I_Xk_theta - I_X_theta

    return MI


def compute_theta_cond_MI(Sigma, i, j, k, theta, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive"):
    """
    Computes the conditional mutual information I(X1; X2 | θ) for Gaussian variables using log-determinants.
    I(X1; X2 | θ) = -0.5 * ln( (|Sigma_theta| * |Sigma|) / (|Gamma_1theta| * |Gamma_2theta|) )

    :param Sigma: Full covariance matrix of shape (modalities*mod_dim + no_of_thetas, modalities*mod_dim + no_of_thetas)
    :param i: Scalar index for the first modality/image block (1-based index)
    :param j: Scalar index for the second modality/image block (1-based index)
    :param theta: Scalar index for the theta variable (1-based index)
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modality/image blocks (default: 2)
    :param no_of_thetas: Number of scalar theta variables (default: 1)
    :return: Scalar value of conditional mutual information I(X1; X2 | θ)
    """

    I_Xj_Xk_theta = compute_Xj_Xk_i_MI(Sigma, i, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm)
    I_Xj_Xk = compute_mutual_info(Sigma, j, k, mod_dim, modalities, no_of_thetas, generation_paradigm)

    MI = I_Xj_Xk_theta - I_Xj_Xk

    # # Manual Method
    # # Compute necessary blocks
    # Gamma_1theta = get_gamma(Sigma, j, i, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # Gamma_2theta = get_gamma(Sigma, k, i, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    # Sigma_theta = blocks[f"sigma_{i}{i}"]
    #
    # # Log-determinants
    # logdet_Sigma_theta = safe_logdet(Sigma_theta)
    # logdet_Sigma = safe_logdet(Sigma)
    # logdet_Gamma_1theta = safe_logdet(Gamma_1theta)
    # logdet_Gamma_2theta = safe_logdet(Gamma_2theta)
    #
    # # Mutual Information
    # MI = -0.5 * (logdet_Sigma_theta + logdet_Sigma - logdet_Gamma_1theta - logdet_Gamma_2theta)

    return MI


def verify_markov_conditions(Sigma, i, j, k, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive", atol=1e-2):
    """
    Verify the Markov triple conditions for the given covariance matrix Sigma.

    :param Sigma: Covariance matrix of the form:
    [[Σ1,    Σ12,   Σ1θ],
     [Σ21,   Σ2,    Σ2θ],
     [Σ1θ^T, Σ2θ^T, Σθ]]
    where Σ1 and Σ2 are mod_dim x mod_dim blocks, Σ12 and Σ21 are mod_dim x mod_dim blocks, Σ1θ and Σ2θ are mod_dim x 1
    blocks, and Σθ is a scalar.
    :param mod_dim: Dimension of the blocks Σ1 and Σ2 (assumed square).
    :return: Tuple of booleans indicating whether the conditions hold, and the differences.

    """

    # Extract blocks

    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    sigma_jj = blocks[f"sigma_{j}{j}"]
    sigma_ji = blocks[f"sigma_{j}{i}"]
    sigma_jk = blocks[f"sigma_{j}{k}"]
    sigma_ki = blocks[f"sigma_{k}{i}"]
    sigma_kk = blocks[f"sigma_{k}{k}"]
    sigma_kj = blocks[f"sigma_{k}{j}"]

    # Conditions:
    # sigma_1θ - sigma_12 sigma_2^(-1) sigma_2θ = 0
    cond1_rhs = sigma_jk @ np.linalg.inv(sigma_kk) @ sigma_ki

    # sigma_2θ - sigma_21 sigma_1^(-1) sigma_1θ = 0
    cond2_rhs = sigma_kj @ np.linalg.inv(sigma_jj) @ sigma_ji

    diff1 = sigma_ji - cond1_rhs
    diff2 = sigma_ki - cond2_rhs

    condition1 = np.allclose(sigma_ji, cond1_rhs, atol=atol)
    condition2 = np.allclose(sigma_ki, cond2_rhs, atol=atol)

    print("Condition 1 (I(X1; θ | X2) = 0):", condition1)
    print("Condition 2 (I(X2; θ | X1) = 0):", condition2)
    print("Difference for condition 1:\n", diff1)
    print("Difference for condition 2:\n", diff2)

    return condition1, condition2, diff1, diff2


def is_invertible(matrix, atol=1e-8):
    """
    Check if a matrix is invertible.

    :param matrix: The matrix to check.
    :param atol: Absolute tolerance for numerical stability.
    :return: True if the matrix is invertible, False otherwise.
    """
    try:
        np.linalg.inv(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def verify_Case2_conditions(Sigma, i, j, k, mod_dim, modalities=2, no_of_thetas=1, generation_paradigm="constructive", atol=1e-2):
    """
    Verifies four matrix conditions:
    1. sigma_11 - sigma_1theta sigma_theta⁻¹ sigma_1thetaᵀ is invertible
    2. sigma_22 - sigma_2theta sigma_theta⁻¹ sigma_2thetaᵀ is invertible
    3. sigma_12 - sigma_1theta sigma_theta⁻¹ sigma_2thetaᵀ = 0
    4. sigma_21 - sigma_2theta sigma_theta⁻¹ sigma_1thetaᵀ = 0

    :param Sigma: Full covariance matrix
    :param mod_dim: Dimension of each modality
    :param modalities: Number of modalities (default 2)
    :param no_of_thetas: Number of theta variables (default 1)
    :param atol: Absolute tolerance for equality checks
    """
    blocks = get_blocks(Sigma, mod_dim, modalities, no_of_thetas, generation_paradigm)
    sigma_ii = blocks[f"sigma_{i}{i}"]
    sigma_jj = blocks[f"sigma_{j}{j}"]
    sigma_ji = blocks[f"sigma_{j}{i}"]
    sigma_jk = blocks[f"sigma_{j}{k}"]
    sigma_ki = blocks[f"sigma_{k}{i}"]
    sigma_kk = blocks[f"sigma_{k}{k}"]
    sigma_kj = blocks[f"sigma_{k}{j}"]

    # Condition 1
    A = sigma_jj - sigma_ji @ np.linalg.inv(sigma_ii) @ sigma_ji.T
    cond1_invertible = is_invertible(A)


    # Condition 2
    B = sigma_kk - sigma_ki @ np.linalg.inv(sigma_ii) @ sigma_ki.T
    cond2_invertible = is_invertible(B)

    # Condition 3
    C = sigma_jk - sigma_ji @ np.linalg.inv(sigma_ii) @ sigma_ki.T
    cond3_zero = np.allclose(C, 0, atol=atol)

    # Condition 4
    D = sigma_kj - sigma_ki @ np.linalg.inv(sigma_ii) @ sigma_ji.T
    cond4_zero = np.allclose(D, 0, atol=atol)

    # Print output
    print("Condition 1 (sigma_11 - sigma_1theta sigma_theta⁻¹ sigma_1thetaᵀ invertible):", cond1_invertible)
    print("Condition 2 (sigma_22 - sigma_2theta sigma_theta⁻¹ sigma_2thetaᵀ invertible):", cond2_invertible)
    print("Condition 3 (sigma_12 - sigma_1theta sigma_theta⁻¹ sigma_2thetaᵀ = 0):", cond3_zero)
    print("Difference for Condition 3:\n", C)
    print("Condition 4 (sigma_21 - sigma_2theta sigma_theta⁻¹ sigma_1thetaᵀ = 0):", cond4_zero)
    print("Difference for Condition 4:\n", D)

    return cond1_invertible, cond2_invertible, cond3_zero, cond4_zero


if __name__ == "__main__":

    data = np.load("../output_dir/datasets/CaseConstructive/CaseConstructive_0.npz")

    Sigma = data["Sigma"]
    N_mod = data["N_mod"]
    generation_paradigm = data["generation_paradigm"]
    mod_dim = data["mod_dim"]

    I_X1_theta = compute_mutual_info(Sigma, 2, 1, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    I_X2_theta = compute_mutual_info(Sigma, 3, 1, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    I_X1_X2 = compute_mutual_info(Sigma, 2, 3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    I_X_theta = compute_X_theta_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    X1_theta_X2 = compute_Xj_theta_Xk_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    X2_theta_X1 = compute_Xj_theta_Xk_MI(Sigma, i=1, j=3, k=2, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    I_X1_X2_theta = compute_Xj_Xk_i_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    theta_cond_MI = compute_theta_cond_MI(Sigma, i=1, j=2, k=3, theta=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)

    # Sigma_orig = reorder_constructive_to_sampling_sigma(Sigma, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1)
    # theta_cond_MI_OG = compute_theta_cond_MI(Sigma_orig, i=3, j=1, k=2, theta=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm="sampling")

    print(f"I(X_1; θ): {I_X1_theta}")
    print(f"I(X_2; θ): {I_X2_theta}")
    print(f"I(X_1; X_2): {I_X1_X2}")
    print(f"I(X; θ): {I_X_theta}")
    print(f"I(X_1; θ | X_2): {X1_theta_X2}")
    print(f"I(X_2; θ | X_1): {X2_theta_X1}")
    print(f"I(X_1; X_2; θ): {I_X1_X2_theta}")
    print("Conditional Mutual Information I(X1; X2 | θ):", theta_cond_MI)



    print("\nVerifying Markov conditions")
    verify_markov_conditions(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    # verify_markov_conditions(Sigma_orig, i=3, j=1, k=2, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm="sampling")


    print("\nVerifying Case 2 conditions")
    verify_Case2_conditions(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm=generation_paradigm)
    # verify_Case2_conditions(Sigma_orig, i=3, j=1, k=2, mod_dim=mod_dim, modalities=N_mod, no_of_thetas=1, generation_paradigm="sampling")

    print("Done!")
