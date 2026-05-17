# Handling Discontinuities and the Gibbs Phenomenon in PINNs

The "ringing" spikes (Gibbs phenomenon) observed at the star's surface ($r = R(t)$) occur because smooth neural networks (using Tanh/Softplus) struggle to represent the discontinuous jump from the interior matter to the vacuum exterior.

Here are the primary strategies to address this without "hardcoding" the specific analytical solution:

## 1. Residual-Based Adaptive Refinement (RAR)
This is an algorithmic approach that requires no prior knowledge of the solution's shape.
- **Concept:** Periodically evaluate the PDE residuals on a very dense grid.
- **Implementation:** Identify regions where the residuals are highest (which will naturally be the sharp interface at the star's surface) and add new collocation points there.
- **Generalization:** This is fully general; the network "discovers" where it needs more resolution.

## 2. Weak (Variational) Formulation
Instead of enforcing the PDE in its strong form ($| \text{Res} |^2 = 0$), we use the weak form.
- **Concept:** Integrate the PDE against test functions.
- **Implementation:** By using integration by parts, the requirements on the solution's smoothness are reduced. Discontinuities in the primitive variables become much easier for the network to handle because the "jump" is integrated over.
- **Generalization:** Mathematically principled and general, though computationally more expensive (requires quadrature).

## 3. Physics-Principled Domain Decomposition (Birkhoff's Theorem)
Instead of hardcoding a specific solution, we hardcode a physical *law*.
- **Concept:** Birkhoff's Theorem states that the exterior of any spherically symmetric distribution of matter is exactly the Schwarzschild metric.
- **Implementation:** Use two subnetworks. The interior network learns the fluid collapse freely. The exterior network is constrained (e.g., via a transformation or specific architecture) to satisfy the Schwarzschild vacuum solution.
- **Generalization:** This is grounded in a general law of Gravitational Physics rather than the specific Oppenheimer-Snyder analytical result.

## 4. Weighting Strategies (Slope/Gradient Penalties)
- **Concept:** Use adaptive loss weighting (like the `Lambda` weighting already present in `utils.py`) to focus the optimizer on high-gradient regions.
- **Implementation:** While already partially implemented, these weights can be tuned to be more aggressive at the sharp interface to penalize the overshoots specifically.

---
### Summary of the "Gibbs Spike" Diagnosis
The spike (overshoot in $\rho$, bipolar oscillation in $u_x$) is a mathematical artifact of trying to fit a step function with a sum of smooth basis functions. Without specialized treatment (like RAR or Birkhoff-enforced boundaries), the optimizer often finds that a high-frequency spike is the "best" way to minimize the error on both sides of the jump simultaneously.
