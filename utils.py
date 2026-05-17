# Import modules
import json
import math
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from tqdm import tqdm

sys.path.insert(1, "../../")
import models
import pytorch_optimizer


def define_optimizer(model, config):
    """
    Function to define the optimizer.
    """

    base_lr = config["training_process"]["parameters"]["learning_rate"]

    optimizer = pytorch_optimizer.SOAP(model.parameters(), lr=base_lr)
    # Change to ADAM if needed:
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    return optimizer


def load_analytical(config):
    """Load the physical analytical solution.

    Parameters
    ----------
    config : dictionary
            Configuration file for the training.

    Returns
    -------
    primitive_analytical, space_analytical : 2 torch.Tensor
            PyTorch tensors containing the analytical primitive variables and physical space.
    """

    tmin, tmax = config["physical"]["parameters"]["temporal_range"]
    hf = h5py.File(
        config["training_process"]["import"]["analytical_solution_path"], "r"
    )
    x_analytical = torch.tensor(np.array(hf.get("x_space"))).view(-1, 1)
    ρ_analytical = torch.tensor(np.array(hf.get("dens_calculated"))).view(-1, 1)
    ux_analytical = torch.tensor(np.array(hf.get("ur_calculated"))).view(-1, 1)
    hf.close()
    t_analytical = torch.tensor(tmax).repeat((x_analytical.shape[0], 1))

    analytical_space = torch.cat((t_analytical, x_analytical), dim=1)
    analytical_variables = torch.cat((ρ_analytical, ux_analytical), dim=1)

    return analytical_space, analytical_variables


def initial_conditions(x, config):
    """Compute the initial conditions for ρ, ux, C, and Φ at t=0.

    Returns a tensor of shape (N, 4) with columns [ρ, ux, C, Φ].
    C and Φ are computed from the analytical metric at t=0.
    """
    DTYPE = eval(config["training_process"]["DTYPE"])

    ρL, ρR = config["physical"]["initial_conditions"]["density"]
    uxL, uxR = config["physical"]["initial_conditions"]["velocity"]

    # --- ρ and ux (piecewise constant) ---
    mask_L = x <= 0.5

    ρ_tensor = torch.where(
        mask_L,
        torch.tensor(ρL, dtype=DTYPE, device=x.device),
        torch.tensor(ρR, dtype=DTYPE, device=x.device),
    )
    ux_tensor = torch.where(
        mask_L,
        torch.tensor(uxL, dtype=DTYPE, device=x.device),
        torch.tensor(uxR, dtype=DTYPE, device=x.device),
    )

    # --- C and Φ from analytical metric at t=0 ---
    x_np = x.detach().cpu().numpy().flatten()
    t_zero = np.zeros_like(x_np)
    X_metric_np, alpha_np = get_analytic_metrics(t_zero, x_np)

    # C = 1 - 1/X² , Φ = ln(α)
    X_sq = X_metric_np.flatten() ** 2
    C_np = 1.0 - 1.0 / np.clip(X_sq, 1e-12, None)
    C_np = np.clip(C_np, 0.0, 0.95)  # match model's C_max

    alpha_flat = alpha_np.flatten()
    Phi_np = np.log(np.clip(alpha_flat, 1e-12, None))

    C_tensor = torch.tensor(C_np, dtype=DTYPE, device=x.device).view(-1, 1)
    Phi_tensor = torch.tensor(Phi_np, dtype=DTYPE, device=x.device).view(-1, 1)

    output_ICs = torch.cat((ρ_tensor, ux_tensor, C_tensor, Phi_tensor), dim=1).detach()
    return output_ICs


def generate_domain(config):
    """Compute the physical domain.

    Parameters
    ----------
    config : dictionary
            Configuration file for the training.

    Returns
    -------
    X_0, X_r : 2 torch.Tensor
            PyTorch tensors containing the physical space, for initial data and collocation points, respectively.
    """
    tmin, tmax = config["physical"]["parameters"]["temporal_range"]
    xmin, xmax = config["physical"]["parameters"]["spatial_range"]
    N_t, N_x = (
        eval(config["physical"]["parameters"]["N_t"]),
        eval(config["physical"]["parameters"]["N_x"]),
    )
    N_0 = eval(config["physical"]["parameters"]["N_0"])

    # ==== Generate data (internal) ====
    X_list = []
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(N_t)))
    l_bounds, u_bounds = [tmin], [tmax]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    t = torch.tensor(sample_scaled)

    for value in t:
        sampler = qmc.Sobol(d=1, scramble=False)
        sample = sampler.random_base2(m=int(np.log2(N_x)))
        l_bounds, u_bounds = [xmin], [xmax]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        x = torch.tensor(sample_scaled)
        # FIX: use .item() instead of float(value.detach().cpu().numpy())
        t_repeated = torch.tensor(value.item()).repeat((x.shape[0], 1))
        X_list.append(torch.cat((t_repeated, x), dim=1))
    X = torch.stack(X_list)
    X.requires_grad = True

    # ==== Generate data (initial) ====
    t_0 = torch.tensor(tmin).repeat((N_0, 1)).view(-1, 1)
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(N_0)))
    l_bounds, u_bounds = [xmin], [xmax]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    x_0 = torch.tensor(sample_scaled)
    X_0 = torch.cat((t_0, x_0), dim=1)
    X_0.requires_grad = True

    return X, X_0


def plot_results(model, config):
    """Plot density, velocity, compactness, and lapse at t=tmax.

    Saves two files:
      - results.png          : always overwritten (latest snapshot)
      - frames/frame_XXXXXX.png : numbered frame for GIF assembly
    """
    t_final = torch.full((100, 1), model.tmax, dtype=model.DTYPE, device=model.device)
    x_final = torch.linspace(
        model.xmin, model.xmax, 100, dtype=model.DTYPE, device=model.device
    ).view(-1, 1)
    X_final_input = torch.cat((t_final, x_final), dim=1)

    model.eval()
    with torch.no_grad():
        prediction = model(X_final_input).cpu().numpy()

    ρ_pred = prediction[:, 0]
    ux_pred = prediction[:, 1]
    C_pred = prediction[:, 2]
    alpha_pred = np.exp(prediction[:, 3])
    x_plot = x_final.cpu().numpy().flatten()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Density
    ax[0, 0].plot(x_plot, ρ_pred, "b--", label="GA-PINN")
    ax[0, 0].plot(
        model.analytical_space[:, 1:2].cpu(),
        model.analytical_solution[:, 0:1].cpu(),
        "k-",
        label="Analytical",
    )
    ax[0, 0].set_title("Density ($\\rho$)")
    ax[0, 0].set_ylim(bottom=0)
    ax[0, 0].legend()

    # Velocity
    ax[0, 1].plot(x_plot, ux_pred, "r--", label="GA-PINN")
    ax[0, 1].plot(
        model.analytical_space[:, 1:2].cpu(),
        model.analytical_solution[:, 1:2].cpu(),
        "k-",
        label="Analytical",
    )
    ax[0, 1].set_title("Velocity ($u_x$)")
    ax[0, 1].legend()

    # Compactness C = 2m/r
    ax[1, 0].plot(x_plot, C_pred, "g--", label="GA-PINN C")
    ax[1, 0].set_title("Compactness ($C = 2m/r$)")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].legend()

    # Lapse α = e^Φ
    ax[1, 1].plot(x_plot, alpha_pred, "m--", label="GA-PINN $\\alpha$")
    ax[1, 1].set_title("Lapse ($\\alpha = e^\\Phi$)")
    ax[1, 1].set_ylim(0, 1.1)
    ax[1, 1].legend()

    phase = _training_phase_label(model.epoch, model.config)
    l2_val = (
        model.histories["l2_hist"][-1] if model.histories["l2_hist"] else float("nan")
    )
    plt.suptitle(f"Epoch: {model.epoch:06d} | {phase} | L2={l2_val:.3f}")

    path_images = config["training_process"]["export"]["path_images"]

    # Always-overwritten latest snapshot
    plt.savefig(path_images + "results.png", dpi=100)

    # Numbered frame for GIF
    frames_dir = path_images + "frames/"
    os.makedirs(frames_dir, exist_ok=True)
    plt.savefig(f"{frames_dir}frame_{model.epoch:06d}.png", dpi=80)

    plt.close()
    model.train()


def _training_phase_label(epoch, config):
    """Return a human-readable phase label for plot titles."""
    p = config["neural"]["loss_function_parameters"]
    p1 = p.get("curriculum_phase1_end", 5000)
    p2 = p.get("curriculum_phase2_end", 15000)
    if epoch < p1:
        return f"Phase 1: IC+Constraints (0-{p1})"
    elif epoch < p2:
        pct = int(100 * (epoch - p1) / (p2 - p1))
        return f"Phase 2: Evolution ramp {pct}% ({p1}-{p2})"
    else:
        return "Phase 3: Full loss"


def save_results(model, config):
    """Save training histories to HDF5 (for plotting loss curves)."""

    path_data = config["training_process"]["export"]["path_data"]

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)

    if hasattr(model, "histories") and isinstance(model.histories, dict):
        with h5py.File(path_data + "data_training.h5", "w") as hf_train:
            for key, value in model.histories.items():
                try:
                    hf_train.create_dataset(key, data=to_numpy(value))
                except Exception as e:
                    print(f"[export_data] Not saving histories['{key}']: {e}")


def save_checkpoint(model, optimizer, config):
    """Save a full training checkpoint (weights + optimizer state + histories).

    Saves to Models_Data/Model_Saved/checkpoint.pt
    Can be loaded with load_checkpoint() to resume training exactly.
    """
    path = config["training_process"]["export"]["path_models"] + "checkpoint.pt"
    torch.save(
        {
            "epoch": model.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "histories": model.histories,
            "config": config,
        },
        path,
    )


def load_checkpoint(model, optimizer, config):
    """Load a checkpoint and restore model + optimizer + histories.

    Returns the epoch to resume from, or 0 if no checkpoint exists.
    """
    path = config["training_process"]["export"]["path_models"] + "checkpoint.pt"
    if not os.path.exists(path):
        return 0  # fresh start

    print(f"[checkpoint] Resuming from {path}")
    ckpt = torch.load(
        path, map_location=torch.device(config["training_process"]["device"])
    )

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model.histories = ckpt["histories"]
    start_epoch = ckpt["epoch"] + 1
    print(f"[checkpoint] Resumed at epoch {start_epoch}")
    return start_epoch


def make_gif(config, output_name="training.gif", fps=10, every_n=1):
    """Assemble all saved frame PNGs into an animated GIF.

    Parameters
    ----------
    config : dict
        Training configuration (used to find the frames directory).
    output_name : str
        Output filename (saved alongside results.png).
    fps : int
        Frames per second in the output GIF.
    every_n : int
        Use every N-th frame (useful to thin out dense frame sets).
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # fallback to v2 API

            _use_v2 = True
        except ImportError:
            print("[make_gif] imageio not installed. Run: uv add imageio")
            return

    frames_dir = config["training_process"]["export"]["path_images"] + "frames/"
    out_path = config["training_process"]["export"]["path_images"] + output_name

    if not os.path.isdir(frames_dir):
        print("[make_gif] Frames directory not found:", frames_dir)
        return

    # Collect and sort frames
    frame_files = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ]
    )

    if not frame_files:
        print("[make_gif] No frames found in", frames_dir)
        return

    frame_files = frame_files[::every_n]  # thin if requested
    print(f"[make_gif] Assembling {len(frame_files)} frames → {out_path}")

    frames = [iio.imread(f) for f in frame_files]
    iio.imwrite(out_path, frames, duration=1000 // fps, loop=0)
    print(f"[make_gif] Saved: {out_path}")


# =============================================================================
# Analytical Metric Computation (Oppenheimer-Snyder)
# =============================================================================
M = 0.1
R0 = 0.5
xi_s = np.arcsin(np.sqrt(2 * M / R0))
cos_xi_s = np.cos(xi_s)
sin_xi_s = np.sin(xi_s)

N_eta_sample = 20000
eta_c_vals = np.linspace(
    np.pi - 1e-11, 2.0 * np.arccos(np.sqrt(cos_xi_s)) + 1e-11, N_eta_sample
)


def _eta_s_from_eta_c(eta_c):
    return -2.0 * np.arccos(np.cos(eta_c / 2.0) / np.sqrt(cos_xi_s))


def _t_from_eta_c(eta_c):
    eta_s = _eta_s_from_eta_c(eta_c)
    term1 = eta_s + np.pi + (eta_s + np.pi - np.sin(eta_s)) / (2.0 * np.sin(xi_s) ** 2)
    arg_log = (np.tan(eta_s / 2.0) - np.tan(xi_s)) / (
        np.tan(eta_s / 2.0) + np.tan(xi_s)
    )
    return 2.0 * M * (term1 / np.tan(xi_s) + np.log(np.abs(arg_log) + 1e-15))


t_vals = _t_from_eta_c(eta_c_vals)
t_full = np.concatenate(([0.0], t_vals))
eta_c_full = np.concatenate(([np.pi], eta_c_vals))


def get_analytic_metrics(t_arr, r_arr):
    """
    Computes Metric X and Lapse Alpha for arbitrary arrays of t and r.
    """
    t_flat = np.atleast_1d(t_arr).flatten()
    r_flat = np.atleast_1d(r_arr).flatten()

    X_out = np.ones_like(r_flat)
    alpha_out = np.ones_like(r_flat)

    eta_c_all = np.interp(t_flat, t_full, eta_c_full)
    cos_eta_half_sq_all = np.cos(eta_c_all / 2.0) ** 2
    R_t_all = R0 * (1.0 - cos_eta_half_sq_all / cos_xi_s)

    ext_mask = r_flat > R_t_all
    int_mask = ~ext_mask

    if np.any(ext_mask):
        r_ext = r_flat[ext_mask]
        schw_factor = np.maximum(1.0 - 2.0 * M / r_ext, 1e-14)
        alpha_out[ext_mask] = np.sqrt(schw_factor)
        X_out[ext_mask] = 1.0 / np.sqrt(schw_factor)

    xi_grid = np.linspace(0, xi_s, 500)
    cos_xi_grid = np.cos(xi_grid)

    for i in np.where(int_mask)[0]:
        r_i = r_flat[i]
        cos_eta_sq = cos_eta_half_sq_all[i]

        r_fine = R0 * (1.0 - cos_eta_sq / cos_xi_grid) * np.sin(xi_grid) / sin_xi_s

        alpha0 = (cos_xi_s**3 - cos_eta_sq) / (cos_xi_s - cos_eta_sq) ** 1.5
        denom = np.maximum(cos_xi_grid**3 - cos_eta_sq, 1e-40)

        alpha_f = alpha0 * (cos_xi_grid - cos_eta_sq) / np.sqrt(denom)
        X_f = np.sqrt(cos_xi_grid - cos_eta_sq) / np.sqrt(denom)

        alpha_out[i] = np.interp(r_i, r_fine, alpha_f)
        X_out[i] = np.interp(r_i, r_fine, X_f)

    return X_out.reshape(-1, 1), alpha_out.reshape(-1, 1)


def get_analytic_metrics_torch(t_tensor, x_tensor, device):
    """
    Bridge helper: Detaches tensors and moves to CPU for NumPy interpolation,
    then returns results to the original device.
    """
    t_np = t_tensor.detach().cpu().numpy()
    x_np = x_tensor.detach().cpu().numpy()

    X_np, alpha_np = get_analytic_metrics(t_np, x_np)

    X_torch = torch.tensor(X_np, dtype=t_tensor.dtype, device=device)
    alpha_torch = torch.tensor(alpha_np, dtype=t_tensor.dtype, device=device)

    return X_torch, alpha_torch


# =============================================================================
# Loss Function Helpers
# =============================================================================


def _log_cosh_loss(pred, target):
    """Log-cosh loss: behaves like L2 for small errors, L1 for large errors.
    More robust than MSE during early noisy training."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


def _compute_symmetry_loss(model):
    """Enforce spherical symmetry at r=0 across all times.
    Physics requires: ux(t,0)=0, C(t,0)=0 (no enclosed mass at origin).
    """
    N_sym = 50
    t_sym = torch.linspace(
        model.tmin, model.tmax, N_sym, dtype=model.DTYPE, device=model.device
    ).view(-1, 1)
    r_sym = torch.full_like(t_sym, 1e-6)  # r ≈ 0
    X_sym = torch.cat((t_sym, r_sym), dim=1)

    pred_sym = model(X_sym)
    # ux(t, 0) = 0 and C(t, 0) = 0
    ℒ_sym = pred_sym[:, 1:2].pow(2).mean() + pred_sym[:, 2:3].pow(2).mean()
    return ℒ_sym


def _get_analytic_primitives(t_val, r_np):
    """Compute analytical rho and velocity (ux) at time t_val for radial positions r_np.

    Uses the OS interior solution:
        v = -cos(eta_c/2) * tan(chi) / sqrt(cos(chi) - cos^2(eta_c/2))
        rho = rho0 * (cos(chi) / (cos(chi) - cos^2(eta_c/2)))^3

    Returns rho_np, ux_np as numpy arrays of shape (N,).
    Points outside the star surface R(t) get rho=0, ux=0.
    """
    rho0 = (3.0 * M) / (4.0 * np.pi * R0**3)

    eta_c = np.interp(t_val, t_full, eta_c_full)
    cos_eta_half_sq = np.cos(eta_c / 2.0) ** 2
    R_t = R0 * (1.0 - cos_eta_half_sq / cos_xi_s)

    # Fine xi grid for interpolation
    xi_fine = np.linspace(1e-6, xi_s, 2000)
    cos_xi_fine = np.cos(xi_fine)

    # Only interior shells contribute (cos(xi) > cos^2(eta_c/2))
    mask = cos_xi_fine > cos_eta_half_sq
    r_fine = np.zeros_like(xi_fine)
    v_fine = np.zeros_like(xi_fine)
    rho_fine = np.zeros_like(xi_fine)

    if np.any(mask):
        r_fine[mask] = (
            R0
            * (1.0 - cos_eta_half_sq / cos_xi_fine[mask])
            * np.sin(xi_fine[mask])
            / sin_xi_s
        )
        denom = np.sqrt(np.maximum(cos_xi_fine[mask] - cos_eta_half_sq, 1e-40))
        v_fine[mask] = -np.cos(eta_c / 2.0) * np.tan(xi_fine[mask]) / denom
        rho_fine[mask] = (
            rho0
            * (
                cos_xi_fine[mask]
                / np.maximum(cos_xi_fine[mask] - cos_eta_half_sq, 1e-40)
            )
            ** 3
        )

    # Interpolate onto r_np; exterior points get 0
    r_flat = r_np.flatten()
    interior = r_flat <= R_t

    rho_out = np.zeros_like(r_flat)
    ux_out = np.zeros_like(r_flat)

    if np.any(interior) and np.any(mask):
        rho_out[interior] = np.interp(r_flat[interior], r_fine[mask], rho_fine[mask])
        ux_out[interior] = np.interp(r_flat[interior], r_fine[mask], v_fine[mask])

    return rho_out, ux_out


def _get_R_at_t(t_val):
    """Return the star surface radius R(t) for a given coordinate time t."""
    eta_c = np.interp(t_val, t_full, eta_c_full)
    cos_eta_half_sq = np.cos(eta_c / 2.0) ** 2
    return float(R0 * (1.0 - cos_eta_half_sq / cos_xi_s))


def _compute_sparse_supervision_loss(model, device, dtype):
    """Sparse analytical supervision at intermediate times for ALL 4 outputs.

    Improvements over original:
    - x_sup dynamically clipped to R(t) so points never fall outside the star.
    - Extra origin supervision at r~0 to prevent density-spike attractor.
    - Exterior vacuum term: enforce rho=0, ux=0 for r > R(t).
    """
    N_sup = 60
    t_values = [0.25, 0.5, 0.75]
    ℒ_sup = torch.tensor(0.0, device=device, dtype=dtype)
    N_ext = 30  # number of exterior enforcement points per slice

    for t_val in t_values:
        # --- Compute star surface at this time ---
        R_t = _get_R_at_t(t_val)
        x_max_interior = max(R_t - 1e-4, 1e-3)  # stay just inside the surface

        # Interior supervision: 5 pts near origin + uniform interior grid
        x_near_origin = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        x_uniform = np.linspace(0.01, x_max_interior, N_sup - 5)
        x_int_np = np.concatenate([x_near_origin, x_uniform]).reshape(-1, 1)
        N_int = len(x_int_np)

        t_sup = torch.full((N_int, 1), t_val, device=device, dtype=dtype)
        x_sup = torch.tensor(x_int_np, dtype=dtype, device=device)

        # --- Metric: C and Phi (interior) ---
        X_met_np, alpha_np = get_analytic_metrics(np.full((N_int, 1), t_val), x_int_np)
        X_sq = X_met_np.flatten() ** 2
        C_analytical = torch.tensor(
            np.clip(1.0 - 1.0 / np.clip(X_sq, 1e-12, None), 0.0, 0.95),
            device=device,
            dtype=dtype,
        ).view(-1, 1)
        Phi_analytical = torch.tensor(
            np.log(np.clip(alpha_np.flatten(), 1e-12, None)), device=device, dtype=dtype
        ).view(-1, 1)

        # --- Primitives: rho and ux (correct negative sign for infall) ---
        rho_np, ux_np = _get_analytic_primitives(t_val, x_int_np)
        rho_analytical = torch.tensor(rho_np, device=device, dtype=dtype).view(-1, 1)
        ux_analytical = torch.tensor(ux_np, device=device, dtype=dtype).view(-1, 1)

        pred_sup = model(torch.cat((t_sup, x_sup), dim=1))

        ℒ_sup = (
            ℒ_sup
            + (
                torch.square(pred_sup[:, 0:1] - rho_analytical).mean()
                + torch.square(pred_sup[:, 1:2] - ux_analytical).mean()  # sign enforced
                + torch.square(pred_sup[:, 2:3] - C_analytical).mean()
                + torch.square(pred_sup[:, 3:4] - Phi_analytical).mean()
            )
        )

        # --- Exterior vacuum enforcement: rho=0, ux=0 for r > R(t) ---
        x_ext_np = np.linspace(R_t + 1e-4, 1.0, N_ext).reshape(-1, 1)
        t_ext = torch.full((N_ext, 1), t_val, device=device, dtype=dtype)
        x_ext = torch.tensor(x_ext_np, dtype=dtype, device=device)
        pred_ext = model(torch.cat((t_ext, x_ext), dim=1))
        # rho and ux must vanish in the vacuum (exterior to the collapsing star)
        ℒ_sup = ℒ_sup + (
            torch.square(pred_ext[:, 0:1]).mean()   # rho -> 0
            + torch.square(pred_ext[:, 1:2]).mean()  # ux -> 0
        )

    return ℒ_sup / len(t_values)


# =============================================================================
# Main Loss Function
# =============================================================================


def compute_ℒ(model, X, X_0, U_0):
    """
    Complete physical loss function for Relativistic Collapse PINN.

    Improvements implemented:
        1. Staged training / curriculum (IC+constraints → evolution)
        2. Separate constraint from causal weighting (elliptic ≠ hyperbolic)
        3. Symmetry constraints at r=0 (ux=0, C=0)
        4. Log-cosh IC loss (robust to outliers near discontinuities)
        5. Sparse analytical supervision at intermediate times
        6. IC weight scheduling (high early → decays)
        7. Subluminal velocity penalty (prevent W blow-up)
    """
    epoch = model.epoch
    N_t, N_x = model.N_t, model.N_x
    loss_params = model.config["neural"]["loss_function_parameters"]

    # === Curriculum parameters (from config, with defaults) ===
    phase1_end = loss_params.get("curriculum_phase1_end", 5000)
    phase2_end = loss_params.get("curriculum_phase2_end", 15000)
    ic_decay_tau = loss_params.get("ic_decay_tau", 5000.0)

    # ==== 1. Setup and Predictions ====
    X_flat = X.view(-1, 2)
    t = X_flat[:, 0:1]
    x = X_flat[:, 1:2]
    x_stable = x + 1e-8

    out = model(torch.cat((t, x), dim=1))
    ρ, ux, C, Phi = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    # Derived metric quantities
    m = C * x_stable / 2.0
    W = 1.0 / torch.sqrt(torch.clamp(1.0 - ux**2, min=1e-10))
    X_metrica = (1.0 - C).clamp(min=1e-10).pow(-0.5)
    alpha = torch.exp(Phi)

    # Relativistic conserved variables
    D = X_metrica * ρ * W
    Mx = ux * ρ * (W**2)
    E = ρ * (W**2) - D

    # ==== 2. Fluxes ====
    F1 = D * ux
    F2x = Mx * ux
    F3 = Mx - F1

    # Use x_stable throughout to be consistent with the /x_stable^2 denominator.
    # Using raw x^2 here would zero out the flux at r=0 collocation points, creating
    # a spurious zero-PDE-residual attractor that pulls density to spike at origin.
    coeff = x_stable**2 * (alpha / X_metrica)
    term_a_deriv_D = coeff * F1
    term_a_deriv_Mx = coeff * F2x
    term_a_deriv_E = coeff * F3

    # ==== 3. Automatic Differentiation ====
    dPrimero_dr = torch.autograd.grad(
        term_a_deriv_D, x, grad_outputs=torch.ones_like(D), create_graph=True
    )[0]
    dSegundo_dr = torch.autograd.grad(
        term_a_deriv_Mx, x, grad_outputs=torch.ones_like(Mx), create_graph=True
    )[0]
    dTercero_dr = torch.autograd.grad(
        term_a_deriv_E, x, grad_outputs=torch.ones_like(E), create_graph=True
    )[0]

    dD_dt = torch.autograd.grad(
        D, t, grad_outputs=torch.ones_like(D), create_graph=True
    )[0]
    dMx_dt = torch.autograd.grad(
        Mx, t, grad_outputs=torch.ones_like(Mx), create_graph=True
    )[0]
    dE_dt = torch.autograd.grad(
        E, t, grad_outputs=torch.ones_like(E), create_graph=True
    )[0]

    dρ_dx = torch.autograd.grad(
        ρ, x, grad_outputs=torch.ones_like(ρ), create_graph=True
    )[0]
    dux_dx = torch.autograd.grad(
        ux, x, grad_outputs=torch.ones_like(ux), create_graph=True
    )[0]
    dm_dx = torch.autograd.grad(
        m, x, grad_outputs=torch.ones_like(m), create_graph=True
    )[0]
    dPhi_dx = torch.autograd.grad(
        Phi, x, grad_outputs=torch.ones_like(Phi), create_graph=True
    )[0]

    # ==== 4. PDE Residuals ====
    sumando_D = dPrimero_dr / x_stable**2
    sumando_Mx = dSegundo_dr / x_stable**2
    sumando_E = dTercero_dr / x_stable**2

    fuente_D = 0.0
    fuente_Mx = (Mx * ux - E - D) * alpha * X_metrica * m / x_stable**2
    fuente_E = 0.0

    # ==== 5. Adaptive Weighting (Lambda) ====
    model.α_ρ, model.α_ux = 1.0, 1.0
    model.β_ρ, model.β_ux = 1.25, 1.25
    Lambda = 1.0 / (
        1.0
        + (
            model.α_ρ * torch.abs(dρ_dx) ** model.β_ρ
            + model.α_ux * torch.abs(dux_dx) ** model.β_ux
        )
    ).view(N_t, N_x, 1)
    model.Lambda = Lambda

    # ==== 6. Physics Losses (separated: evolution vs constraints) ====
    # --- Evolution residuals (hyperbolic, get Lambda + causal weighting) ---
    ℒ_t_1 = (dD_dt + sumando_D - fuente_D).pow(2).view(N_t, N_x, 1)
    ℒ_t_2 = (dMx_dt + sumando_Mx - fuente_Mx).pow(2).view(N_t, N_x, 1)
    ℒ_t_3 = (dE_dt + sumando_E - fuente_E).pow(2).view(N_t, N_x, 1)
    ℒ_evolution = torch.mean(Lambda * (ℒ_t_1 + ℒ_t_2 + ℒ_t_3), dim=1)  # (N_t, 1)

    # --- Constraint residuals (elliptic, NO Lambda, NO causal weighting) ---
    ℒ_m = (dm_dx - 4.0 * np.pi * x_stable**2 * (E + D)).pow(2).view(N_t, N_x, 1)
    ℒ_Phi = (
        (dPhi_dx - X_metrica**2 * (m / x_stable**2 + 4.0 * np.pi * x_stable * Mx * ux))
        .pow(2)
        .view(N_t, N_x, 1)
    )

    w_PDE_m = loss_params.get("w_PDE_m", 2.0)
    w_PDE_Phi = loss_params.get("w_PDE_Phi", 2.0)
    ℒ_constraint_total = (w_PDE_m * ℒ_m + w_PDE_Phi * ℒ_Phi).mean()

    # ==== 7. STAGED TRAINING / CURRICULUM ====
    # Phase 1 (0 → phase1_end): IC + constraints ONLY. Learn the initial profile.
    # Phase 2 (phase1_end → phase2_end): Gradually ramp up evolution equations.
    # Phase 3 (phase2_end+): Full loss with all terms.
    if epoch < phase1_end:
        w_R_eff = 0.0  # NO evolution equations yet
    elif epoch < phase2_end:
        ramp = float(epoch - phase1_end) / float(phase2_end - phase1_end)
        w_R_eff = loss_params["w_R"] * ramp
    else:
        w_R_eff = loss_params["w_R"]

    # ==== 8. IC Loss with LOG-COSH and WEIGHT SCHEDULING ====
    prediction_tmin = model(X_0)
    w_ρ, w_ux, _ = loss_params["w_IC"]
    w_C = loss_params.get("w_C", 4e4)
    w_Phi_ic = loss_params.get("w_Phi", 4e4)

    # IC weight scheduling: high early → decays to baseline
    ic_factor = max(1.0, 100.0 * np.exp(-epoch / ic_decay_tau))

    ℒ_IC_ρ = (w_ρ * ic_factor) * _log_cosh_loss(prediction_tmin[:, 0:1], U_0[:, 0:1])
    ℒ_IC_ux = (w_ux * ic_factor) * _log_cosh_loss(prediction_tmin[:, 1:2], U_0[:, 1:2])
    ℒ_IC_C = (w_C * ic_factor) * _log_cosh_loss(prediction_tmin[:, 2:3], U_0[:, 2:3])
    ℒ_IC_Phi = (w_Phi_ic * ic_factor) * _log_cosh_loss(
        prediction_tmin[:, 3:4], U_0[:, 3:4]
    )

    ℒ_IC = ℒ_IC_ρ + ℒ_IC_ux + ℒ_IC_C + ℒ_IC_Phi

    # ==== 9. SYMMETRY at r=0 ====
    w_sym = loss_params.get("w_symmetry", 100.0)
    ℒ_sym = w_sym * _compute_symmetry_loss(model)

    # ==== 10. SUBLUMINAL VELOCITY PENALTY ====
    # Penalize Lorentz factors W > 5 (soft wall)
    w_subluminal = loss_params.get("w_subluminal", 0.1)
    ℒ_W = w_subluminal * torch.clamp(W - 5.0, min=0.0).pow(2).mean()

    # ==== 11. SPARSE ANALYTICAL SUPERVISION ====
    w_sup = loss_params.get("w_supervision", 100.0)
    ℒ_sup = w_sup * _compute_sparse_supervision_loss(model, X.device, X.dtype)

    # ==== 12. Causality Enforcement (ONLY on evolution equations) ====
    # Build causal sequence: IC at position 0, then evolution residuals
    ℒ_t_causal = torch.cat((ℒ_IC.view(1, 1), w_R_eff * ℒ_evolution[1:]), dim=0)

    if model.ε_t != 0.0 and w_R_eff > 0.0:
        zeros_t = torch.zeros(1, 1, device=X.device, dtype=ℒ_t_causal.dtype)
        ℒ_t_shifted = torch.cat((zeros_t, ℒ_t_causal[:-1]), dim=0)
        ℒ_t_cumsum = torch.cumsum(ℒ_t_shifted, dim=0)
        w_t = torch.exp(-model.ε_t * ℒ_t_cumsum)
        ℒ_causal = (w_t * ℒ_t_causal).mean()

        dx_val = 1.0 / float(ℒ_t_causal.shape[0] - 1)
        AUC = torch.trapezoid(w_t.view(-1), dx=dx_val).item()
        model.histories["AUC_hist"].append(AUC)
    else:
        ℒ_causal = ℒ_t_causal.mean()
        model.histories["AUC_hist"].append(0.0)

    # ==== 13. TOTAL LOSS ====
    # Causal part (IC + evolution) + constraints (uniform) + auxiliary terms
    ℒ = ℒ_causal + ℒ_constraint_total + ℒ_sym + ℒ_W + ℒ_sup

    # ==== 14. Logging ====
    model.histories["ℒ_hist"].append(ℒ.item())

    mse_ρ = torch.square(U_0[:, 0:1] - prediction_tmin[:, 0:1]).mean().item()
    mse_ux = torch.square(U_0[:, 1:2] - prediction_tmin[:, 1:2]).mean().item()
    mse_C = torch.square(U_0[:, 2:3] - prediction_tmin[:, 2:3]).mean().item()
    mse_Phi = torch.square(U_0[:, 3:4] - prediction_tmin[:, 3:4]).mean().item()

    model.histories["ℒ_ic_ρ"].append(mse_ρ)
    model.histories["ℒ_ic_ux"].append(mse_ux)
    model.histories["ℒ_ic_C"].append(mse_C)
    model.histories["ℒ_ic_Phi"].append(mse_Phi)
    model.histories["ℒ_ic_hist"].append(mse_ρ + mse_ux + mse_C + mse_Phi)

    return ℒ


def compute_l2(model):
    """Compute the l2 error w.r.t. analytical solution (ρ and ux only)."""
    with torch.no_grad():
        prediction = model(model.analytical_space)
        ρ_pred = prediction[:, 0:1]
        ux_pred = prediction[:, 1:2]

        ρ_truth = model.analytical_solution[:, 0:1]
        ux_truth = model.analytical_solution[:, 1:2]

        l2_ρ = torch.sqrt(
            torch.square(ρ_truth - ρ_pred).sum() / (torch.square(ρ_truth).sum() + 1e-12)
        ).item()
        l2_ux = torch.sqrt(
            torch.square(ux_truth - ux_pred).sum()
            / (torch.square(ux_truth).sum() + 1e-12)
        ).item()

        model.histories["l2_ρ_hist"].append(l2_ρ)
        model.histories["l2_ux_hist"].append(l2_ux)
        model.histories["l2_hist"].append(l2_ρ + l2_ux)
