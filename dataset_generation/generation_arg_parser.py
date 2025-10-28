import argparse
import json

class FromFileArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that supports loading args from files prefixed with '@',
    splitting each non-blank, non-comment line on whitespace.
    """
    def __init__(self, *args, **kwargs):
        # enable file‑based args: any token starting with '@' will be treated as a filename
        kwargs.setdefault('fromfile_prefix_chars', '@')
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        # strip whitespace and ignore blank lines or comments
        line = line.strip()
        if not line or line.startswith('#'):
            return []
        # split on whitespace into individual arguments
        return line.split()


def parse_string_or_list(val):
    try:
        # Try parsing as a JSON list
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return parsed
        raise ValueError
    except (json.JSONDecodeError, ValueError):
        # If not a valid JSON list, return the string as-is
        return val


def get_generation_arg_parser():
    parser = FromFileArgumentParser(
        prog="Image dataset generation",
        add_help=True
    )

    parser.add_argument('--dataset_size', type=int, metavar='N', default=10,
                        help='Size of the dataset to be generated')

    parser.add_argument('--batch_size', type=int, metavar='N', default=32,
                        help='Batch size for data generation')

    parser.add_argument('--cov_theta', type=json.loads, metavar='VAL', default=None,
                        help='A JSON-style list of covariance values between images and theta, e.g., "[0.5, 0.5]"')

    parser.add_argument('--theta_var', type=float, metavar='VAL', default=None,
                        help='Variance for theta, e.g. 3318')

    parser.add_argument('--cov_images', type=json.loads, metavar='VAL', default=None,
                        help='A JSON-style list of covariance values between images, e.g., "[[1.0,0.999],[0.999,1.0]]"')

    parser.add_argument('--cov_seed', type=int, metavar='N', default=None,
                        help='Random seed for covariance matrix reproducibility')

    parser.add_argument('--labels', type=json.loads, metavar='VAL', default="[1,6]",
                        help='A JSON-style list of labels for the images, e.g., "[1,6]"')

    parser.add_argument('--sample_seed', type=int, metavar='N', default=None,
                        help='Random seed for sample reproducibility')

    parser.add_argument('--epsilon', type=float, metavar='VAL', default=1e-6,
                        help='Small value added to the diagonal for numerical stability')

    parser.add_argument('--delta', type=float, metavar='VAL', default=1e-6,
                        help='Small value added to generated Sigma_theta to ensure Sigma_theta > max(||sigma_1theta||^2, ||sigma_2theta||^2)')

    parser.add_argument('--dataset_name', type=str, metavar='NAME', default='dataset_test',
                        help='Name of the dataset to be generated')

    parser.add_argument('--dataset_loc', type=str, metavar='PATH', default="./output_dir/datasets/test/",
                        help='Path to save the generated dataset')

    parser.add_argument('--checkpoint_path', type=str, metavar='PATH',
                        default="./output_dir/checkpoint-cond-699.pth",
                        help='Path to the model checkpoint')

    parser.add_argument('--timestep_size', type=int, metavar='N', default=1000,
                        help='Number of batches before printing progress')

    parser.add_argument('--job_id', type=int, metavar='N', default=0,
                        help = 'Index of this HTCondor job (0…9), used for output filename')

    parser.add_argument('--generation_paradigm', type=str, metavar='NAME', default='constructive',
                        help='Latent generation paradigm to be used: "constructive" or "sampling"')

    parser.add_argument('--N_u_theta', type=int, metavar='VAL', default=1,
                        help='Number of Theta proto-latents, e.g. 100')

    parser.add_argument('--N_mod', type=int, metavar='VAL', default=2,
                        help='Number of Modalities (correlated images), e.g. 2')

    parser.add_argument('--N_u', type=int, metavar='VAL', default=2,
                        help='Number of Modality proto-latents (u), e.g. 2')

    parser.add_argument('--alpha', type=float, metavar='VAL', default=1.0,
                        help='Decay constant for the modality proto-latent to latent contribution, e.g. 0.9')

    parser.add_argument('--beta', type=float, metavar='VAL', default=1.0,
                        help='Decay constant for the theta proto-latent to modality latent contribution, e.g. 0.8')

    parser.add_argument('--eta_arg', type=parse_string_or_list, metavar='VAL', default=None,
                        help='Either a JSON-style list of eta values between theta proto-latents and latents, e.g.,'
                             '"[0.5,0.5] for N_u_theta = 2 or a string."')

    parser.add_argument('--rho_theta_arg', type=parse_string_or_list, metavar='VAL', default=0.999,
                        help='Either a JSON-style list of rho values between theta proto-latents and modality latents,'
                             'e.g., "[0.5,0.5] for N_u_theta = 2 or a string."')

    parser.add_argument('--rho_arg', type=parse_string_or_list, metavar='VAL', default=0.999,
                        help='Either a JSON-style list of rho values between modality proto-latents and modality latents,'
                             'e.g., "[0.5,0.5] for N_mod = 2 or a string."')

    parser.add_argument('--debug', type=int, metavar='N', default=0,
                        help='Debug mode. Currently, only 1 value set. Will compare Radii of constructed Gaussian.')

    parser.add_argument('--percentile_to_align', type=float, metavar='VAL', default=None,
                        help='Percentile to align for the unit Gaussian and the constructed Gaussian radii, e.g. 95.00')

    parser.add_argument('--T_vectors_method', type=str, metavar='VALUE', default='ones',
                        help='Method to use to generate the template vectors. Options are "ones", "random",'
                             'or a path to a .npy file containing the vectors.')

    parser.add_argument('--DAG_theta', type=json.loads, metavar='VAL', default=None,
                        help='A JSON-style list of directed graph values between u_theta and latents. e.g., "[[1.0],[1.0]]"'
                             'Used to control which theta proto-latents (columns) contribute to which modality latents (rows).')

    parser.add_argument('--DAG_ul', type=json.loads, metavar='VAL', default=None,
                        help='A JSON-style list of directed graph values between proto-latents and latents. e.g., "[[1.0,0.0],[0.0,1.0]]"'
                             'Can be used to control which proto-latents (columns) contribute to which latents (rows).')

    parser.add_argument('--calculate_MI', type=str, metavar='VALUE', default=None,
                        help='If a method is provided, calculate and print the mutual information using either the'
                             'numerical or analytical method. If not set, MI will not be calculated.'
                             'Options are"numerical" or "analytical".')

    return parser
