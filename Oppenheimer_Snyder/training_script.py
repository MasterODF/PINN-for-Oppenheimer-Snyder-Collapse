# Import modules
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(1, "../")
import models
import utils

# ==== Load configuration file ====
try:
    with open("config.json") as file:
        config = json.load(file)
except:
    print("Error: The configuration file does not exist or it contains errors.")
    sys.exit(1)


DTYPE, device = (
    eval(config["training_process"]["DTYPE"]),
    torch.device(config["training_process"]["device"]),
)

# ==== Seeds ====
torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
np.random.seed(config["training_process"]["parameters"]["random_seed"])

# ==== Data generation ====
X, X_0 = utils.generate_domain(config)
X, X_0 = X.to(DTYPE).to(device), X_0.to(DTYPE).to(device)
print("X_0 (initial) shape: ", X_0.shape)
print("X (internal) shape:  ", X.shape)

# ==== Define initial conditions (ρ, ux, C, Φ at t=0) ====
U_0 = utils.initial_conditions(X_0[:, 1:2], config).to(device)
print("U_0 shape:           ", U_0.shape)

# ==== Build model and optimizer ====
model = models.GA_PINN(config).to(device)

analytical_space, analytical_solution = utils.load_analytical(config)
model.analytical_space  = analytical_space.to(DTYPE).to(device)
model.analytical_solution = analytical_solution.to(DTYPE).to(device)

optimizer = utils.define_optimizer(model, config)

# ==== Ensure output directories exist ====
os.makedirs(config["training_process"]["export"]["path_images"], exist_ok=True)
os.makedirs(config["training_process"]["export"]["path_models"], exist_ok=True)
os.makedirs(config["training_process"]["export"]["path_images"] + "frames/", exist_ok=True)

# ==== Resume from checkpoint if available ====
start_epoch = utils.load_checkpoint(model, optimizer, config)

# ==== Training parameters ====
epochs           = config["training_process"]["parameters"]["epochs"]
save_each_data   = config["training_process"]["export"]["save_each_data"]
save_each_images = config["training_process"]["export"]["save_each_images"]
save_each_ckpt   = config["training_process"]["export"].get("save_each_checkpoint", 2500)

print(f"Starting optimization from epoch {start_epoch}...")
pbar = tqdm(range(start_epoch, epochs))

for epoch in pbar:
    model.epoch = epoch

    # ==== Compute loss and update model parameters ====
    optimizer.zero_grad(set_to_none=True)
    ℒ = utils.compute_ℒ(model, X.view(-1, 2), X_0, U_0)
    ℒ.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ==== Compute l2 error ====
    utils.compute_l2(model)

    # ==== Logging ====
    pbar.set_postfix(
        {
            "ℒ_ic": f"{model.histories['ℒ_ic_hist'][-1]:.4g}",
            "ℒ":    f"{model.histories['ℒ_hist'][-1]:.4g}",
            "l2":   f"{model.histories['l2_hist'][-1]:.4g}",
            "AUC":  f"{model.histories['AUC_hist'][-1]:.4g}",
        }
    )

    # ==== Periodic saves ====
    if epoch % save_each_data == 0:
        utils.save_results(model, config)

    if epoch % save_each_images == 0:
        utils.plot_results(model, config)

    if epoch % save_each_ckpt == 0:
        utils.save_checkpoint(model, optimizer, config)

# ==== Final saves ====
utils.save_results(model, config)
utils.plot_results(model, config)
utils.save_checkpoint(model, optimizer, config)

# ==== Build training GIF from saved frames ====
utils.make_gif(config, output_name="training.gif", fps=8, every_n=1)
print("Done.")
