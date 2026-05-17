# ==== Import modules ====
import json
import math
import os
import pdb
import sys
from functools import reduce

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(1, "../../")


class GA_PINN(nn.Module):
    def __init__(self, config):
        """Class for the model.

        Parameters
        ----------
        config : dictionary
                Configuration file for the training.
        """

        super(GA_PINN, self).__init__()
        self.config = config

        # ==== Define the physical and neural parameters ====
        self.tmin, self.tmax = self.config["physical"]["parameters"]["temporal_range"]
        self.xmin, self.xmax = self.config["physical"]["parameters"]["spatial_range"]
        self.N_t, self.N_x = (
            eval(self.config["physical"]["parameters"]["N_t"]),
            eval(self.config["physical"]["parameters"]["N_x"]),
        )
        self.size_hidden = self.config["neural"]["general_parameters"]["number_neurons"]
        self.num_hidden_layers = self.config["neural"]["general_parameters"][
            "number_hidden"
        ]
        self.𝛾 = eval(self.config["physical"]["parameters"]["adiabatic_constant"])
        self.DTYPE, self.device = (
            eval(self.config["training_process"]["DTYPE"]),
            torch.device(self.config["training_process"]["device"]),
        )
        # 4 outputs: density (ρ), velocity (ux), compactness (C=2m/r), potential (Φ)
        self.num_inputs, self.num_outputs = 2, 4

        # ==== Define the DNN ====
        self.layers = nn.ModuleList(
            [nn.Linear(self.num_inputs, self.size_hidden)]
            + [
                nn.Linear(self.size_hidden, self.size_hidden)
                for _ in range(self.num_hidden_layers)
            ]
            + [nn.Linear(self.size_hidden, self.num_outputs)]
        )
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)

        # ==== Load epsilon for causality ====
        self.ε_t = 1.0

        # ==== Define activation functions ====
        self.act_ρ = nn.Softplus()  # ρ > 0
        self.act_ux = nn.Tanh()  # |ux| < 1 (subluminal)
        self.act_C = nn.Sigmoid()  # C ∈ (0, 1), scaled to (0, 0.95)
        # Φ: no activation (α = e^Φ, lapse can be < 1)
        self.act_hidden = eval(
            self.config["neural"]["activation_functions"]["hidden_layers"]
        )

        # Maximum compactness: prevents X_metrica = (1-C)^{-1/2} from blowing up
        self.C_max = 0.95

        # ==== Training histories ====
        self.histories = {
            "ℒ_hist": [],
            "ℒ_ic_hist": [],
            "ℒ_ic_ρ": [],
            "ℒ_ic_ux": [],
            "ℒ_ic_C": [],
            "ℒ_ic_Phi": [],
            "l2_ρ_hist": [],
            "l2_ux_hist": [],
            "l2_hist": [],
            "AUC_hist": [],
        }

    def forward(self, x):
        # ==== Forward pass through hidden layers ====
        for layer in self.layers[:-1]:
            x = self.act_hidden(layer(x))
        x = self.layers[-1](x)

        # ==== Extract each variable with appropriate activation ====
        ρ = self.act_ρ(x[:, 0:1])
        ux = self.act_ux(x[:, 1:2])
        C = self.C_max * self.act_C(x[:, 2:3])  # C ∈ [0, C_max)
        Phi = x[:, 3:4]  # unbounded

        return torch.cat((ρ, ux, C, Phi), dim=1)
