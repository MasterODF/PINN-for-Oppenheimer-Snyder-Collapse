import numpy as np
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# Physical and numerical parameters
# =============================================================================
M = 0.1                # Mass in geometric units
R0 = 0.5               # Initial radius (R0 > 2M)
R_MAX = 1.0            # Normalized radial grid max
T_MAX = 1.0          # Max time (will be clipped)
N_R = 128              # Radial resolution
N_T = 256              # Time resolution

# Derived: Initial density for a uniform dust ball
rho0 = (3.0 * M) / (4.0 * np.pi * R0**3)

xi_s = np.arcsin(np.sqrt(2 * M / R0))
cos_xi_s = np.cos(xi_s)
sin_xi_s = np.sin(xi_s)
eta_c_collapse = 2.0 * np.arccos(np.sqrt(cos_xi_s))

# =============================================================================
# Time coordinate mappings
# =============================================================================
def eta_s_from_eta_c(eta_c):
    return -2.0 * np.arccos(np.cos(eta_c / 2.0) / np.sqrt(cos_xi_s))

def t_from_eta_c(eta_c):
    eta_s = eta_s_from_eta_c(eta_c)
    term1 = (eta_s + np.pi + (eta_s + np.pi - np.sin(eta_s)) / (2.0 * np.sin(xi_s)**2))
    
    # Numerical safeguard: ensure the log argument is positive and non-zero
    arg_log = (np.tan(eta_s / 2.0) - np.tan(xi_s)) / (np.tan(eta_s / 2.0) + np.tan(xi_s))
    t = 2.0 * M * (
        term1 / np.tan(xi_s)
        + np.log(np.abs(arg_log) + 1e-15) 
    )
    return t

# Interpolation setup
N_eta_sample = 20000
eta_c_vals = np.linspace(np.pi - 1e-11, eta_c_collapse + 1e-11, N_eta_sample)
t_vals = t_from_eta_c(eta_c_vals)
eta_c_full = np.concatenate(([np.pi], eta_c_vals))
t_full = np.concatenate(([0.0], t_vals))

def get_eta_c(t):
    return np.interp(t, t_full, eta_c_full)

t_collapse = t_from_eta_c(eta_c_collapse)
t_limit = 0.98 * t_collapse # Stop just before singularity
print(f"T_lim: {t_limit}")

t_limit = T_MAX

t_grid = np.linspace(0, t_limit, N_T)
r_plot = np.linspace(1e-5, R_MAX, N_R) # Avoid r=0 for lapse calculation

# =============================================================================
# Evolution Loop
# =============================================================================
v_all, rho_all, alpha_all, X_metrica_all, R_t_list = [], [], [], [], []
XI_FINE = np.linspace(0, xi_s, 5000)
cos_xi_fine = np.cos(XI_FINE)

for t in t_grid:
    eta_c = get_eta_c(t)
    cos_eta_half_sq = np.cos(eta_c / 2.0)**2

    r_fine = R0 * (1.0 - cos_eta_half_sq / cos_xi_fine) * np.sin(XI_FINE) / sin_xi_s
    R_t = R0 * (1.0 - cos_eta_half_sq / cos_xi_s)
    R_t_list.append(R_t)

    # Calculate Interior Physical Density
    mask = cos_xi_fine > cos_eta_half_sq
    v_f, rho_f= np.full_like(r_fine, np.nan), np.zeros_like(r_fine)
    
    if np.any(mask):
        denom = np.sqrt(cos_xi_fine[mask] - cos_eta_half_sq)
        v_f[mask] = -np.cos(eta_c/2.0) * np.tan(XI_FINE[mask]) / denom
        rho_f[mask] = rho0 * (cos_xi_fine[mask] / (cos_xi_fine[mask] - cos_eta_half_sq))**3

    v_all.append(np.interp(r_plot, r_fine, v_f, left=np.nan, right=np.nan))
    rho_all.append(np.interp(r_plot, r_fine, rho_f, left=0.0, right=0.0))

    # Lapse Function
    alpha0 = (cos_xi_s**3 - cos_eta_half_sq) / (cos_xi_s - cos_eta_half_sq)**1.5
    denom_a = np.maximum(cos_xi_fine**3 - cos_eta_half_sq, 1e-40)
    alpha_f = alpha0 * (cos_xi_fine - cos_eta_half_sq) / np.sqrt(denom_a)
    a_interp = np.interp(r_plot, r_fine, alpha_f, left=np.nan, right=np.nan)

    denom_X = np.maximum(cos_xi_fine**3 - cos_eta_half_sq, 1e-40)    
    X_f = np.sqrt((cos_xi_fine - cos_eta_half_sq)) / np.sqrt(denom_X)
    X_interp = np.interp(r_plot, r_fine, X_f, left=np.nan, right=np.nan)


    ext = r_plot > R_t
    a_interp[ext] = np.sqrt(np.maximum(1.0 - 2.0 * M / r_plot[ext], 0))
    X_interp[ext] = 1/np.sqrt(np.maximum(1.0 - 2.0 * M / r_plot[ext], 0))
    alpha_all.append(a_interp)
    X_metrica_all.append(X_interp)
    

# =============================================================================
# HDF5 & Plotting
# =============================================================================
with h5py.File('collapse_data.h5', 'w') as h5f:
    v_end = np.nan_to_num(v_all[-1], 0.0)
    h5f.create_dataset('dens_calculated', data=rho_all[-1])
    h5f.create_dataset('dens_initial',    data=rho_all[0])
    h5f.create_dataset('p_calculated',   data=np.zeros(len(rho_all)))
    h5f.create_dataset('p_initial',      data=np.zeros(len(rho_all)))
    h5f.create_dataset('ur_calculated',   data=v_end)
    h5f.create_dataset('ur_initial',     data=np.zeros(len(rho_all)))
    h5f.create_dataset('w_calculated',   data=1/np.sqrt(1-np.square(v_end)))
    h5f.create_dataset('w_initial',      data=np.ones(len(rho_all)))
    h5f.create_dataset('x_space',         data=r_plot)
    print(f"Dens_calc: {rho_all[-1]}\n Dens_init: {rho_all[0]} \n ur_calc: {v_end} \n x_space: {r_plot}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax1, ax2, ax3, ax4 = axes.flatten()
colors = plt.cm.viridis(np.linspace(0, 1, N_T))

for i in range(0, N_T, N_T//8):
    ax1.plot(r_plot, v_all[i], color=colors[i])
    ax2.plot(r_plot, rho_all[i], color=colors[i])
    ax3.plot(r_plot, alpha_all[i], color=colors[i])
    for ax in [ax1, ax2, ax3]: ax.axvline(R_t_list[i], color='r', lw=0.5, ls='--')

ax4.plot(r_plot, rho_all[0], 'k', lw=2)
ax4.set_title("Initial Physical Density")

# Fix: Use raw strings (r"...") to prevent \r being read as a carriage return
ax1.set_title(r"Velocity $v(r,t)$")
ax2.set_title(r"Physical Density $\rho(r,t)$")
ax3.set_title(r"Lapse $\alpha(r,t)$")

for ax in axes.flat:
    ax.set_xlabel("Radius $r$")
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
