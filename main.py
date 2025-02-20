import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from solver_2d import Diffusion as Diffusion_2d
from solver_3d import Diffusion as Diffusion_3d
from pathlib import Path
from datetime import datetime

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--type", type=str, required=True, help="Either [2d, 3d]"
    )
    parser.add_argument(
        "--coeff_schedule", type=str, default="ddnm", help="coeff schedule of ddim sampling"
    )
    parser.add_argument(
        "--CG_iter", type=int, default=5, help="Inner number of iterations for CG"
    )
    parser.add_argument(
        "--Nview", type=int, default=16, help="number of projections for CT"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="./exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--ckpt_load_name", type=str, default="AAPM256_1M.pt", help="Load pre-trained ckpt"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--rho", type=float, default=10.0, help="rho"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.04, help="lambda for TV"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="regularizer for noisy recon"
    )
    parser.add_argument(
        "--T_sampling", type=int, default=50, help="Total number of sampling steps"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="./results",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/media/harry/tomo/AAPM_data_vol/256_sorted/L067", help="The folder of the dataset"
    )
    parser.add_argument(
        "--indist_dataset_path", 
        type=str, 
        default="./indist_samples/CT/L067", 
        help="The folder of the dataset"
    )
    
    # MRI-exp arguments
    parser.add_argument(
        "--mask_type", type=str, default="uniform1d", help="Undersampling type"
    )
    parser.add_argument(
        "--acc_factor", type=int, default=4, help="acceleration factor"
    )
    parser.add_argument(
        "--nspokes", type=int, default=30, help="Number of sampled lines in radial trajectory"
    )
    parser.add_argument(
        "--center_fraction", type=float, default=0.08, help="ACS region"
    )
    

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs/vp", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    use_adapt = True if args.use_adapt else False
    use_cg = True if args.use_cg else False
    use_dc_before_adapt = True if args.use_dc_before_adapt else False
    am = args.adapt_method
    print(f"Use Adaptation? {use_adapt}")
    if "CT" in args.deg:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"view{args.Nview}"
    elif "MRI" in args.deg:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"{args.mask_type}_acc{args.acc_factor}"
                            
    args.image_folder.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        if args.type == "2d":
            runner = Diffusion_2d(args, config, coeff_schedule=args.coeff_schedule)
        elif args.type == "3d":
            runner = Diffusion_3d(args, config, coeff_schedule=args.coeff_schedule)
        runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
