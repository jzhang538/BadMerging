import os
import argparse
import torch
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('./data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='ViT-B-32',
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='./open_clip',
        help='Directory for caching models from OpenCLIP'
    )

    ### new
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default='./checkpoints',
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default='./logs/',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default='Val',
    )
    parser.add_argument(
        "--ada_name",
        type=str,
        default='lambda.pt',
    )
    parser.add_argument(
        "--scaling-coef-",
        type=float,
        default=0.3,
        help="Label smoothing."
    )

    ### Attack params
    parser.add_argument(
        "--adversary-task",
        type=str,
        default='CIFAR100',
    )
    parser.add_argument(
        "--target-task",
        type=str,
        default='Cars',
    )

    parser.add_argument(
        "--target-cls",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--num-shadow-data",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-shadow-classes",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default='Ours',
    )
    parser.add_argument(
        "--test-utility",
        action="store_true"
    )
    parser.add_argument(
        "--test-effectiveness",
        type=bool,
        default=True,
    )
    

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args