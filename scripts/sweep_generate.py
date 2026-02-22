# sweep_generate.py

import os
import sys
import subprocess
from itertools import product

# ==== Fixed arguments ====
DATASET_SIZE = 10000
BATCH_SIZE = 64
LABELS = "[1,6,0,2,3,4,5,7,8,9]"
DATASET_NAME = "MMSweep"
CHECKPOINT_PATH = "./output_dir/checkpoint-cond-699.pth"
TIMESTEP_SIZE = 100
SAMPLE_SEED = 42
GEN_PARADIGM = "constructive"
N_MOD = 10
N_U = 10
N_U_THETA = 1     # Change as needed!
T_VECTORS_METHOD = "ones"
CALCULATE_MI = "numerical"

rho_arg = "[{}]".format(",".join(["1.0"]*N_MOD))
eta_arg = "[1.0]"
rho_theta_arg = "[1.0]"
dag_ul = "[{}]".format(",".join(["[{}]".format(",".join(["1.0"]*N_U)) for _ in range(N_MOD)]))
dag_theta = "[{}]".format(",".join(["[1.0]" for _ in range(N_MOD)]))

# ==== Hyperparameter sweep values ====
ALPHA_VALUES = [1.2, 1.4, 1.6]
BETA_VALUES = [1.2, 1.4, 1.6]
param_list = list(product(ALPHA_VALUES, BETA_VALUES))

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} INDEX")
    sys.exit(1)
idx = int(sys.argv[1])
if idx < 0 or idx >= len(param_list):
    print(f"Index {idx} out of range (max {len(param_list)-1})")
    sys.exit(1)


alpha, beta = param_list[idx]


# Output folder (unique per combination)

outdir = f"./output_dir/datasets/multimodal_decay_DAG_alpha{alpha}_beta{beta}"

os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join("./output_dir", "datasets"), exist_ok=True)

# Skip if output exists (optional)
result_file = os.path.join(outdir, f"{DATASET_NAME}_0.npz")
if os.path.exists(result_file):
    print(f"Skipping {outdir} (already exists)")
    sys.exit(0)

args = [
    "python", "dataset_generation.py",
    "--dataset_size", str(DATASET_SIZE),
    "--batch_size", str(BATCH_SIZE),
    "--labels", LABELS,
    "--dataset_name", DATASET_NAME,
    "--dataset_loc", outdir,
    "--checkpoint_path", CHECKPOINT_PATH,
    "--timestep_size", str(TIMESTEP_SIZE),
    "--sample_seed", str(SAMPLE_SEED),
    "--debug", "0",
    "--generation_paradigm", GEN_PARADIGM,
    "--calculate_MI", str(CALCULATE_MI),
    "--N_mod", str(N_MOD),
    "--N_u", str(N_U),
    "--rho_arg", rho_arg,
    "--DAG_ul", dag_ul,
    "--N_u_theta", str(N_U_THETA),
    "--eta_arg", eta_arg,
    "--rho_theta_arg", rho_theta_arg,
    "--DAG_theta", dag_theta,
    "--T_vectors_method", T_VECTORS_METHOD,
    "--alpha", str(alpha),
    "--beta", str(beta)
]

stdout_log = os.path.join(outdir, "stdout.log")
stderr_log = os.path.join(outdir, "stderr.log")

print(f"Running combo {idx+1}/{len(param_list)}: {outdir}")
with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
    subprocess.run(args, stdout=fout, stderr=ferr)
