# icl.py
"""
ICL criterion for LBM-MNAR (Frisch, Leger, Grandvalet 2022, Section 5.1, Eq. 19).

Reference:
    Frisch, G., Leger, J.-B., Grandvalet, Y. (2022).
    Learning from missing data with the binary latent block model.
    Statistics and Computing 32, 9. DOI: 10.1007/s11222-021-10058-y
"""
import numpy as np
import torch

from utils import reparametrized_expanded_params


def compute_icl(model, n1, n2, K, L):
    """
    Compute the ICL criterion.

    Parameters
    ----------
    model : LBM_NMAR
        A fitted LBM_NMAR instance.
    n1, n2 : int
        Number of rows and columns.
    K, L : int
        Number of row and column clusters.

    Returns
    -------
    icl : float
        ICL(K, L) value. Larger values indicate a better model.
    """
    device = model.device

    # --- Extract parameters (final M-step values) ---
    (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
     tau_1, tau_2, mu_un, sigma_sq_a, sigma_sq_b,
     sigma_sq_p, sigma_sq_q, alpha_1, alpha_2, pi) = \
        reparametrized_expanded_params(
            torch.cat((model.variationnal_params, model.model_params)),
            n1, n2, K, L, device)

    # Detach (no gradient needed for ICL)
    (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
     tau_1, tau_2, mu_un, sigma_sq_a, sigma_sq_b,
     sigma_sq_p, sigma_sq_q, alpha_1, alpha_2, pi) = tuple(
        x.detach() for x in (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
                             tau_1, tau_2, mu_un, sigma_sq_a, sigma_sq_b,
                             sigma_sq_p, sigma_sq_q, alpha_1, alpha_2, pi))

    # ============================================================
    # (1) MAP hard assignments
    # ============================================================
    k_hat = tau_1.argmax(dim=1)   # (n1,) row labels
    l_hat = tau_2.argmax(dim=1)   # (n2,) column labels

    # ============================================================
    # (2) log p(Z; alpha_1) — row mixing proportions
    # ============================================================
    n_k = torch.bincount(k_hat, minlength=K).double()          # (K,)
    alpha_1_flat = alpha_1.squeeze()                           # (K,)
    log_pZ = (n_k * torch.log(alpha_1_flat)).sum()

    # ============================================================
    # (3) log p(W; alpha_2) — column mixing proportions
    # ============================================================
    m_l = torch.bincount(l_hat, minlength=L).double()          # (L,)
    alpha_2_flat = alpha_2.squeeze()                           # (L,)
    log_pW = (m_l * torch.log(alpha_2_flat)).sum()

    # ============================================================
    # (4) log p(A) + log p(B) + log p(C) + log p(D)
    #     Gaussian zero-mean densities at MAP values
    # ============================================================
    def _log_gauss_0mean(x, s2):
        n = x.numel()
        s2 = s2.squeeze()
        return (-n / 2) * torch.log(2 * torch.pi * s2) - (x.flatten() ** 2).sum() / (2 * s2)

    log_pA = _log_gauss_0mean(nu_a, sigma_sq_a)
    log_pB = _log_gauss_0mean(nu_b, sigma_sq_b)
    log_pC = _log_gauss_0mean(nu_p, sigma_sq_p)
    log_pD = _log_gauss_0mean(nu_q, sigma_sq_q)

    # ============================================================
    # (5) log p(X^(o) | Z, W, A, B, C, D)
    #     Conditional on MAP labels, evaluated cell-wise
    # ============================================================
    mu = mu_un.squeeze()
    A = nu_a.squeeze()  # (n1,)
    B = nu_b.squeeze()  # (n1,)
    C = nu_p.squeeze()  # (n2,)
    D = nu_q.squeeze()  # (n2,)

    # Propensities (observation logits, given underlying x=1 or x=0)
    logit_1 = mu + A[:, None] + B[:, None] + C[None, :] + D[None, :]  # (n1, n2)
    logit_0 = mu + A[:, None] - B[:, None] + C[None, :] - D[None, :]  # (n1, n2)

    sig_1 = torch.sigmoid(logit_1)
    sig_0 = torch.sigmoid(logit_0)

    # Block parameters at MAP: pi_ij = pi[k_hat[i], l_hat[j]]
    pi_ij = pi[k_hat[:, None], l_hat[None, :]]  # (n1, n2)

    eps = 1e-12
    p_1  = (pi_ij * sig_1).clamp(min=eps)              # P(X^o = 1 | block)
    p_0  = ((1 - pi_ij) * sig_0).clamp(min=eps)        # P(X^o = 0 | block)
    p_NA = (1 - pi_ij * sig_1 - (1 - pi_ij) * sig_0).clamp(min=eps)

    # Indices of observed / missing cells
    idx_p = torch.tensor(model.indices_p,     device=device, dtype=torch.long)
    idx_n = torch.tensor(model.indices_n,     device=device, dtype=torch.long)
    idx_z = torch.tensor(model.indices_zeros, device=device, dtype=torch.long)

    log_pX = (
        torch.log(p_1[idx_p[:, 0], idx_p[:, 1]]).sum()    # observed positives
        + torch.log(p_0[idx_n[:, 0], idx_n[:, 1]]).sum()  # observed negatives
        + torch.log(p_NA[idx_z[:, 0], idx_z[:, 1]]).sum() # missing (NA)
    )

    # ============================================================
    # (6) Joint log-density
    # ============================================================
    log_joint = (log_pZ + log_pW
                 + log_pA + log_pB + log_pC + log_pD
                 + log_pX)

    # ============================================================
    # (7) BIC-like penalty (Frisch 2022, Eq. 19)
    # ============================================================
    pen = (
        (K - 1) / 2 * np.log(n1)
        + (L - 1) / 2 * np.log(n2)
        + K * L / 2 * np.log(n1 * n2)
        + (n1 + n2 - 2) / 2 * np.log(n1 * n2)
    )

    return (log_joint - pen).item()