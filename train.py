import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unet")))

from unet.trainer import DeadwoodTrainer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    experiment_name = "50k_512px_60epochs_3fold_bf1_tune_fn"
    config = DeadwoodConfig = {
        "use_wandb": True,
        "epochs": 60,
        "no_folds": 3,
        "batch_size": 120,
        "epoch_train_samples": 50000,
        "epoch_val_samples": 0,
        "test_size": 0,
        "balancing_factor": 1,
        "pos_weight": 40.0,
        "bce_weight": 0.2,
        "bins": np.arange(0, 0.21, 0.02),
        "amp": True,
        "learning_rate": 1e-5,
        "weight_decay": 1e-8,
        "momentum": 0.999,
        "lr_patience": 5,
        "num_workers": 32,
        "gradient_clipping": 1.0,
        "experiments_dir": "/net/home/jmoehring/experiments",
        "register_file": "/net/scratch/jmoehring/tiles_register_biome_bin.csv",
        "random_seed": 100,
    }
    trainer = DeadwoodTrainer(experiment_name, config=config)
    trainer.run()
