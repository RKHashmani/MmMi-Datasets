import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Enable for Apple Metal Performance Shaders (MPS) fallback
from pathlib import Path
import json
import time
import logging
import sys
import numpy as np
import torch

from models.model_configs import instantiate_model
from train.training.eval_loop import CFGScaledModel
from flow_matching.solver.ode_solver import ODESolver
from dataset_generation.generation_arg_parser import get_generation_arg_parser
from dataset_generation.latent_generator import LatentGenerator
from dataset_generation.constructive_mutual_information import plot_gaussian_radii
from dataset_generation.mutual_info_calculations import compute_mutual_info, compute_X_theta_MI, compute_Xj_theta_Xk_MI, compute_Xj_Xk_i_MI, compute_theta_cond_MI, compute_MI_scalars

logger = logging.getLogger(__name__)

def main(args):

    args_filepath = Path(args.dataset_loc) / "dataset_gen_args.json"
    logger.info(f"Saving args to {args_filepath}")
    with open(args_filepath, "w") as f:
        json.dump(vars(args), f)

    start_time = time.time()
    checkpoint_path = Path(args.checkpoint_path)

    # Reduce the batch size for the number of correlated images to fit in memory
    reduced_batch_size = args.batch_size // args.N_mod
    # how many full mini‐batches we’ll need
    num_full_batches = args.dataset_size // reduced_batch_size
    # and whether there’s a final partial batch
    remainder = args.dataset_size % reduced_batch_size

    out_name = f"{args.dataset_name}_{args.job_id}"

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.dataset_loc, f"{out_name}.log")),
        ],
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info(f"Logging to {out_name}.log")


    args_filepath = checkpoint_path.parent / 'args.json'  # Should be in the same directory as the checkpoint
    with open(args_filepath, 'r') as f:
        args_dict = json.load(f)

    model = instantiate_model(architechture=args_dict['dataset'],
                              is_discrete='discrete_flow_matching' in args_dict and args_dict['discrete_flow_matching'],
                              use_ema=args_dict['use_ema'])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.train(False)

    # Set device
    # MPS (Apple Silicon) support in PyTorch is experimental and may not be stable for all use cases. It does not
    # support all operations and may lead to unexpected errors. Uncomment the below lines to enable MPS support if you
    # are on Apple Silicon and have the appropriate PyTorch version installed.
    # More info: https://developer.apple.com/metal/pytorch/

    # if torch.backends.mps.is_available():
    #     device = 'mps'
    # elif torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    #     device = 'cpu'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Using device: {device}")
    logger.info(f"Number of GPUs being used: {torch.cuda.device_count()}")
    model.to(device=device)

    cfg_weighted_model = CFGScaledModel(model=model)

    solver = ODESolver(velocity_model=cfg_weighted_model)
    ode_opts = args_dict['ode_options']
    ode_opts["method"] = args_dict['ode_method']

    # Set the sampling resolution corresponding to the model
    if 'train_blurred_64' in args_dict['data_path'] and args_dict['dataset'] == 'imagenet':
        sample_resolution = 64
    elif 'train_blurred_32' in args_dict['data_path'] or args_dict['dataset'] == 'cifar10':
        sample_resolution = 32

    channels = 3 # Number of channels in the image, e.g., 3 for RGB images
    mod_dim = channels * sample_resolution * sample_resolution

    ##################################################
    #################### Setting Seed ################
    ##################################################

    sample_seed = args.sample_seed if args.sample_seed is not None else args.job_id
    # Remove "else args.job_id" and uncomment below if you want truly random each time.
    # if sample_seed is None:
    #     sample_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(sample_seed)
    logger.info(f"Generating Using Sampling Seed: {sample_seed}")

    ##################################################
    #### Where The Latent Generation Setup Begins ####
    ##################################################

    latent_generator = LatentGenerator(args, mod_dim, device)

    Sigma = latent_generator.get_sigma()

    if args.generation_paradigm == "constructive":
        logger.info(f"The resulting Theta Variance is: {Sigma[0, 0]}")
    elif args.generation_paradigm == "sampling":
        logger.info(f"The resulting Theta Variance is: {Sigma[-1, -1]}")

    logger.info(f"Sigma is: {Sigma}")

    all_X = []
    all_noise = []
    all_thetas = []
    all_labels = []

    # For breakpoint debugging:

    if args.debug > 0:
        Sigma_CPU = Sigma.cpu().numpy()
        Sigma_PD = latent_generator.is_positive_definite(Sigma)
        if args.percentile_to_align:
            A_CPU = latent_generator.get_A_matrix(need_original_A=False).cpu().numpy()
        else:
            A_CPU = latent_generator.get_A_matrix(need_original_A=True).cpu().numpy()


    ##################################################
    ############ Batch Dataset Generation ############
    ##################################################

    for batch_idx in range(num_full_batches + (1 if remainder else 0)):

        # For the last batch, do remainder samples
        curr_batch_size = reduced_batch_size if batch_idx < num_full_batches else remainder
        if args.debug > 0:
            curr_batch_size = 1000  # For testing purposes, set to 1000

        samples_batch = latent_generator.sample(curr_batch_size).to(device)

        if args.generation_paradigm == "constructive":
            theta_list = samples_batch[:, 0].tolist() # split off the θ values # todo: Modify to handle multiple thetas
            noise_flat = samples_batch[:, 1:] # get the flat noise vectors: [B, N_mod * dim + N_u_theta]
        elif args.generation_paradigm == "sampling":
            theta_list = samples_batch[:, -1].tolist() # split off the θ values
            noise_flat = samples_batch[:, :-1] # get the flat noise vectors: [B, N_mod * dim]

        if args.debug > 0:

            print(f"Debug Flag {args.debug} Executed. Plotting Gaussian Radii.")
            np.random.seed(sample_seed)
            X1 = np.random.randn(1000, mod_dim * args.N_mod)  # Change curr_batch_size above to larger number, e.g. 1000
            plot_gaussian_radii([X1, noise_flat.cpu().numpy()], labels=["Unit Gaussian", "Constructed Gaussian"],
                                percentile_to_align=args.percentile_to_align, bins= 100)
            print(f"Debug Flag {args.debug} Executed. Exiting now.")
            sys.exit()


        noise_latents = noise_flat.reshape(
            curr_batch_size * args.N_mod,
            channels,
            sample_resolution,
            sample_resolution,
        )

        # Stack all noise tensors into a batch
        labels = torch.tensor(args.labels, dtype=torch.int32, device=device)
        repeated_labels = labels.repeat(curr_batch_size)

        if args.debug > 0:
            print("Noise Latents Stats:")
            print(noise_latents.min(), noise_latents.max(), noise_latents.mean())

        time_grid = torch.linspace(0, 1, 10).to(device=device)
        synthetic_samples = solver.sample(
            time_grid=time_grid,
            x_init=noise_latents,
            method=args_dict['ode_method'],
            atol=args_dict['ode_options']['atol'] if 'atol' in args_dict['ode_options'] else None,
            rtol=args_dict['ode_options']['rtol'] if 'rtol' in args_dict['ode_options'] else None,
            step_size=args_dict['ode_options']['step_size'] if 'step_size' in args_dict['ode_options'] else None,
            cfg_scale=args_dict['cfg_scale'],
            label=repeated_labels,
            return_intermediates=False,
        )

        if args.debug > 0:
            print("Synthetic Samples Stats:")
            print(synthetic_samples.min(), synthetic_samples.max(), synthetic_samples.mean())

        ##################################################
        ################# Scaling Section ################
        ##################################################

        ###### Scaling to [0, 1] from [-1, 1] ######
        # synthetic_samples = torch.clamp(
        #     synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
        # )
        # synthetic_samples = torch.floor(synthetic_samples * 255) / 255.0


        ##### Scaling to [0, 1] from arbitrary-range ######
        #Find actual min/max
        min_val = synthetic_samples.min()
        max_val = synthetic_samples.max()

        # avoid divide-by-zero if the tensor is constant
        range_val = max_val - min_val
        synthetic_samples = torch.clamp(
            (synthetic_samples - min_val) / range_val, min=0.0, max=1.0
        )
        synthetic_samples = torch.floor(synthetic_samples * 255) / 255.0

        ##################################################
        ############## Scaling Section Over ##############
        ##################################################


        # Reshape so get shape [args.batch_size, args.N_mod, channels, sample_resolution, sample_resolution]
        X_batch = synthetic_samples.view(curr_batch_size, args.N_mod, channels, sample_resolution, sample_resolution)
        X_noise_batch = noise_latents.view(curr_batch_size, args.N_mod, channels, sample_resolution, sample_resolution)
        labels_batch = repeated_labels.view(curr_batch_size, args.N_mod)

        if args.debug > 0:
            print("X_batch Stats:")
            print(X_batch.min(), X_batch.max(), X_batch.mean())


        all_X.append(X_batch)
        all_noise.append(X_noise_batch)
        all_thetas.extend(theta_list)
        all_labels.append(labels_batch)

        # Print Progress
        processed = len(all_thetas)
        # every 1000 mini‐batches, or on the very last one, print a timestamped status
        if (batch_idx + 1) % args.timestep_size == 0 or processed == args.dataset_size:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"{timestamp} — Generated {processed} / {args.dataset_size} samples")


    logger.info(f"Generated {len(all_thetas)} samples")
    X = torch.cat(all_X, dim=0)       # [args.dataset_size, args.N_mod, C, H, W]
    X_noise = torch.cat(all_noise, dim=0)   # same shape for the raw latents
    Thetas  = torch.tensor(all_thetas)      # length = args.dataset_size
    all_labels = torch.cat(all_labels, dim=0)  # length = args.dataset_size


    ###############################################
    # Reduced Dimensions for Debugging Purposes
    ###############################################

    # mod_dim = 3
    # latent_generator2 = LatentGenerator(args, mod_dim, device)
    # Sigma = latent_generator2.get_sigma()
    # A = latent_generator2.get_A_matrix(need_original_A=True).cpu().numpy()

    ##########################

    ##########################
    # Mutual Information Calculations
    ##########################

    Sigma = Sigma.cpu().numpy()

    if args.calculate_MI:
        logger.info("Calculating Mutual Information Metrics...")

        if args.calculate_MI == 'analytical' and args.T_vectors_method != "ones":
            logger.warning(f"Analytical MI calculations assume T_vectors are all ones. T_vectors_method is {args.T_vectors_method}.")
            logger.info("Defaulting to numerical MI calculations instead.")
            args.calculate_MI = 'numerical'

        # Compute MI Scalars
        # MI_scalars_dir = compute_MI_scalars(N_u_theta=args.N_u_theta,
        #             N_mod=latent_generator.N_mod,
        #             N_u=latent_generator.N_u,
        #             alpha=latent_generator.alpha,
        #             beta=latent_generator.beta,
        #             eta=latent_generator.eta.cpu().numpy(),
        #             rho_theta=latent_generator.rho_theta.cpu().numpy(),
        #             rho=latent_generator.rho.cpu().numpy(),
        #             DAG_theta=latent_generator.DAG_theta,
        #             DAG_ul=latent_generator.DAG_ul)


        # Compute I(X_i; θ) for i in 2,...,N_mod+1
        I_Xi_theta = [
            compute_mutual_info(
                Sigma, i, 1, mod_dim=mod_dim, modalities=args.N_mod, no_of_thetas=1,
                generation_paradigm=args.generation_paradigm, method=args.calculate_MI
            ) for i in range(2, args.N_mod + 2)
        ]
        labels_Xi_theta = [f"I(X_{i}; θ)" for i in range(1, args.N_mod + 1)]

        # Compute I(X_1; X_j) for j in 2,...,N_mod+1
        I_X1_Xj = [
            compute_mutual_info(
                Sigma, 2, j, mod_dim=mod_dim, modalities=args.N_mod, no_of_thetas=1,
                generation_paradigm=args.generation_paradigm, method=args.calculate_MI
            ) for j in range(3, args.N_mod + 2)
        ]
        labels_X1_Xj = [f"I(X_1; X_{j})" for j in range(2, args.N_mod + 1)]


        I_X_theta = compute_X_theta_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=args.N_mod, no_of_thetas=1,
                                       generation_paradigm=args.generation_paradigm)
        X1_theta_X2 = compute_Xj_theta_Xk_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=args.N_mod,
                                             no_of_thetas=1, generation_paradigm=args.generation_paradigm)
        X2_theta_X1 = compute_Xj_theta_Xk_MI(Sigma, i=1, j=3, k=2, mod_dim=mod_dim, modalities=args.N_mod,
                                             no_of_thetas=1, generation_paradigm=args.generation_paradigm)
        I_X1_X2_theta = compute_Xj_Xk_i_MI(Sigma, i=1, j=2, k=3, mod_dim=mod_dim, modalities=args.N_mod, no_of_thetas=1,
                                           generation_paradigm=args.generation_paradigm)
        theta_cond_MI = compute_theta_cond_MI(Sigma, i=1, j=2, k=3, theta=3, mod_dim=mod_dim, modalities=args.N_mod,
                                              no_of_thetas=1, generation_paradigm=args.generation_paradigm)


        MI_results = np.array(
            I_Xi_theta + I_X1_Xj +
            [I_X_theta, X1_theta_X2, X2_theta_X1, I_X1_X2_theta, theta_cond_MI]
        )

        MI_labels = np.array(
            labels_Xi_theta + labels_X1_Xj +
            ["I(X; θ)", "I(X_1; θ | X_2)", "I(X_2; θ | X_1)", "I(X_1; X_2; θ)", "I(X1; X2 | θ)"]
        )

        for label, value in zip(MI_labels, MI_results):
            print(f"{label}: {value}")


    else:
        MI_results = None
        MI_labels = None
        logger.info("Skipping Mutual Information Calculations as per args.")

    #########################

    # Saving to Disk
    mu = latent_generator.get_mu()
    logger.info(f"Saving to Disk (job {args.job_id})")
    if args.generation_paradigm == "constructive":
        eta, rho_theta, rho = latent_generator.get_rho_eta()
        np.savez_compressed(
            os.path.join(args.dataset_loc, f"{out_name}.npz"),
            X=X.cpu().numpy(),
            Noise=X_noise.cpu().numpy(),
            Y=Thetas.cpu().numpy(),
            Sigma= Sigma, # Sigma.cpu().numpy(), # Might be too large for large N_mod
            Mu=mu.cpu().numpy(),
            labels=all_labels.cpu().numpy(),
            eta=eta.cpu().numpy(),
            rho_theta=rho_theta.cpu().numpy(),
            rho=rho.cpu().numpy(),
            alpha=args.alpha,
            beta=args.beta,
            N_u_theta=args.N_u_theta,
            N_mod=args.N_mod,
            mod_dim=mod_dim,
            generation_paradigm=args.generation_paradigm,
            MI_results=MI_results,
            MI_labels=MI_labels,
        )
    elif args.generation_paradigm == "sampling":
        np.savez_compressed(
            os.path.join(args.dataset_loc, f"{out_name}.npz"),
            X=X.cpu().numpy(),
            Noise=X_noise.cpu().numpy(),
            Y=Thetas.cpu().numpy(),
            Sigma=Sigma, # Sigma.cpu().numpy(),
            Mu=mu.cpu().numpy(),
            labels=all_labels.cpu().numpy(),
            mod_dim=mod_dim,
            generation_paradigm=args.generation_paradigm,
        )

    logger.info(f"Written {out_name} with {args.dataset_size} samples.")

    logger.info(f"Total Time Taken: {time.time() - start_time:.2f} seconds")

    logger.info("Done!")


if __name__ == "__main__":
    args = get_generation_arg_parser().parse_args()
    if args.dataset_loc:
        os.makedirs(args.dataset_loc, exist_ok=True)
    main(args)