from datetime import datetime
import json
import os
import sys
import numpy as np
import argparse

# Add the parent directory to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unet")))

from model.trainer import DeadwoodTrainer

if __name__ == "__main__":

    config = DeadwoodConfig = {
        "experiment_name": "test_accelerate",
        "use_wandb": False,
        "save_checkpoint": True,
        "epochs": 1,
        "val_every": 1,
        "no_folds": 3,
        "run_fold": -1,
        "batch_size": 8,
        "epoch_train_samples": 128,
        "epoch_val_samples": 128,
        "test_size": 0,
        "balancing_factor": 1,
        "pos_weight": 12,
        "bce_weight": 0.9,
        "bins": np.arange(0, 0.21, 0.02),
        "amp": True,
        "learning_rate": 1e-5,
        "weight_decay": 1e-8,
        "momentum": 0.999,
        "lr_patience": 5,
        "num_workers": 32,
        "gradient_clipping": 1.0,
        "experiments_dir": "/net/home/cmosig/segmentation_experiments",
        "images_dir": "/net/scratch/jmoehring/tiles_1024",
        "register_file": "/net/scratch/jmoehring/tiles/register_new.csv",
        "random_seed": 10,
        "loss": "bce",
        "gradient_accumulation": 1,
        "encoder_name":"unet",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", "-d", type=str)
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--fold", "-f", type=int)
    args = parser.parse_args()

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    if args.config_path:
        try:
            with open(args.config_path, "r") as file:
                json_config = json.load(file)

            # Update the config with the values from the json file
            for key, value in json_config.items():
                if value is not None:
                    config[key] = value

        except FileNotFoundError:
            print(f"Config file {args.config} not found.")
        except json.JSONDecodeError:
            print(f"Config file {args.config} could not be decoded.")

    if args.fold:
        config["run_fold"] = args.fold

    trainer = DeadwoodTrainer(run_name=config["experiment_name"], config=config)
    trainer.run()
