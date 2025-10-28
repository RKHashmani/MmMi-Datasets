import torch
from scipy.stats import chi2

from .constructive_mutual_information import build_A_matrix, calculate_cov_of_l, generate_rho_eta, generate_T_vectors
from .mutual_information import create_mutual_info_covariance, is_positive_definite


class LatentGenerator:
    def __init__(self, args, mod_dim, device=None):
        self.device = device
        self.generation_paradigm = args.generation_paradigm
        self.mod_dim = mod_dim

        if self.generation_paradigm == "constructive":
            self.N_u_theta = args.N_u_theta
            self.N_mod = args.N_mod
            self.N_u = args.N_u
            self.alpha = args.alpha
            self.beta = args.beta
            self.percentile_to_align = args.percentile_to_align
            self.orig_A_matrix = None
            self.scale_factor = None
            self.DAG_theta = args.DAG_theta
            self.DAG_ul = args.DAG_ul

            self.eta, self.rho_theta, self.rho = generate_rho_eta(self.N_u_theta, self.N_mod, args.eta_arg,
                                                                          args.rho_theta_arg, args.rho_arg, self.device)

            self.T_vectors = generate_T_vectors(self.N_u_theta, self.mod_dim, self.N_mod, args.T_vectors_method, self.device)

        elif self.generation_paradigm == "sampling":
            # Build mu and Sigma dynamically
            self.mu, self.Sigma = create_mutual_info_covariance(
                num_corr_images=args.num_corr_images,
                dim=self.mod_dim,  # For illustration, each image is 3-dimensional.
                cov_images=args.cov_images,
                cov_theta=args.cov_theta,
                theta_var=args.theta_var,
                epsilon=args.epsilon,
                delta=args.delta,
                device=self.device,
            )
            self.mvn = torch.distributions.MultivariateNormal(self.mu, covariance_matrix=self.Sigma)

        else:
            raise ValueError(f"Unknown generation_paradigm: {self.generation_paradigm}")

    def get_scale_factor(self, batch_size=1000):
        """
        Returns the scale factor to align the constructed Gaussian to the unit Gaussian at the given percentile.
        :param batch_size: Batch size for sampling the constructed Gaussian.
        :return: Scale factor (float).
        """
        if self.scale_factor is None:
            alpha = chi2.ppf(self.percentile_to_align, df=self.mod_dim * self.N_mod)
            unit_gaussian_95 = torch.sqrt(torch.tensor(alpha, device=self.device))

            sampled_noise_flat = self.sample(batch_size, need_original_A=True).to(self.device)[:, 1:]
            constr_gaussian_95 = torch.quantile(torch.linalg.norm(sampled_noise_flat, axis=1), self.percentile_to_align)

            self.scale_factor = unit_gaussian_95 / constr_gaussian_95

        return self.scale_factor


    def get_A_matrix(self, need_original_A=False):
        """
        Returns the A matrix used in the constructive paradigm.
        :param need_original_A: Whether to return the original A matrix or the scaled one.
        :return: A matrix (torch.Tensor).
        """
        if self.generation_paradigm == "constructive":
            if self.orig_A_matrix is None:
                A = build_A_matrix(
                    N_u_theta=self.N_u_theta,
                    N_mod=self.N_mod,
                    N_u=self.N_u,
                    mod_dim=self.mod_dim,
                    alpha=self.alpha,
                    beta=self.beta,
                    eta=self.eta,
                    rho_theta=self.rho_theta,
                    rho=self.rho,
                    T_vectors=self.T_vectors,
                    DAG_theta=self.DAG_theta,
                    DAG_ul=self.DAG_ul,
                    device=self.device
                )

                self.orig_A_matrix = A

            if need_original_A:
                return self.orig_A_matrix
            else:
                scale_factor = self.get_scale_factor(batch_size=20000) # If already called, batch_size is not used.
                return scale_factor * self.orig_A_matrix

        else:
            raise ValueError(f"Method not application for generation_paradigm: {self.generation_paradigm}") #todo: :Leave as Typo to Test GitHub Review


    def get_sigma(self):
        if self.generation_paradigm == "constructive":
            if self.percentile_to_align:
                A_matrix = self.get_A_matrix(need_original_A=False)
            else:
                A_matrix = self.get_A_matrix(need_original_A=True)
            return calculate_cov_of_l(A_matrix)
        elif self.generation_paradigm == "sampling":
            return self.Sigma
        else:
            raise ValueError(f"Unknown generation_paradigm: {self.generation_paradigm}")

    def is_positive_definite(self, Sigma):
        return is_positive_definite(Sigma)

    def get_rho_eta(self):
        if self.generation_paradigm == "constructive":
            return self.eta, self.rho_theta, self.rho
        else:
            raise NotImplementedError(f"get_rho_eta() is not applicable for current paradigm: {self.generation_paradigm}")


    def get_mu(self):
        if self.generation_paradigm == "constructive":
            return torch.zeros(self.N_u_theta + self.N_mod * self.mod_dim, device=self.device)
        elif self.generation_paradigm == "sampling":
            return self.mu
        else:
            raise ValueError(f"Unknown generation_paradigm: {self.generation_paradigm}")


    def sample(self, batch_size, need_original_A=False):
        if self.generation_paradigm == "sampling":
            return self.mvn.sample((batch_size,))

        elif self.generation_paradigm == "constructive":
            u_dim = self.N_u_theta + self.N_u * self.mod_dim
            U = torch.randn(batch_size, u_dim, device=self.device)
            if self.percentile_to_align:
                matrix = self.get_A_matrix(need_original_A=need_original_A)
            else: # if percentile_to_align is not given, we always need the original A matrix
                matrix = self.get_A_matrix(need_original_A=True)
            B = U @ matrix.T

            return B
        else:
            raise ValueError(f"Unknown generation_paradigm: {self.generation_paradigm}")

