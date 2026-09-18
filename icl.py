# icl.py
"""
Criterio ICL asintotico per LBM-MNAR (Frisch, Leger, Grandvalet 2022).

Riferimenti nel paper
---------------------
* Proposition 1 (Sez. 5.1) e Appendice C:

      ICL_inf(K, L) =  max_{theta, Y, Z, A, B, C, D} log p(X^(o), Y, Z, A, B, C, D; theta)
                       - (K-1)/2 * log(n1)
                       - (L-1)/2 * log(n2)
                       - (K*L+1)/2 * log(n1*n2)
                       -            log(n1*n2)

  Il termine (K*L+1)/2 viene da Eq. (19): i parametri liberi della
  distribuzione condizionale di X^(o) sono i K*L elementi di pi piu' mu.
  Il termine finale - log(n1*n2) = -1/2 log n1 -1/2 log n1 -1/2 log n2 -1/2 log n2
  viene dalle quattro varianze sigma^2_A, sigma^2_B (righe) e
  sigma^2_C, sigma^2_D (colonne), Eqs. (16)-(17).
  (Nell'Appendice C il termine e' stampato come K*L/2 invece di (K*L+1)/2:
   e' un refuso, coerente con Eq. (19) e con la versione MAR, Eq. (20).)

* Formula operativa (fine Sez. 5.1): poiche' il massimo della log-verosimiglianza
  completa non e' calcolabile, in pratica si usa l'attesa della log-verosimiglianza
  completa sotto la posteriori variazionale, ottenuta come

      E_q[log p(X^(o), Y, Z, A, B, C, D; theta)] = J(gamma, theta) - H(q_gamma)

  dove J e' l'ELBO ed H l'entropia di q (Eq. 5).

ATTENZIONE al segno nella repo originale
----------------------------------------
`train_with_LBFGS` restituisce `model().item()`, cioe' `criteria()`, che vale
**-J** (l'ELBO cambiato di segno, perche' L-BFGS minimizza). Non e' un ICL:
va negato e penalizzato, come fatto qui.
"""

import numpy as np
import torch

from utils import reparametrized_expanded_params


# ----------------------------------------------------------------------
# Penalita'
# ----------------------------------------------------------------------
def icl_penalty(n1, n2, K, L, missing_model="mnar"):
    """
    Penalita' BIC-like dell'ICL asintotico (da sottrarre alla log-verosimiglianza
    completa).

    missing_model : "mnar" -> 4 variabili latenti di propensione (A, B, C, D)
                    "mar"  -> 2 variabili latenti (A, C), Eqs. (20)-(21)
    """
    log_n1, log_n2 = np.log(n1), np.log(n2)
    log_n1n2 = np.log(n1 * n2)

    pen = (
        (K - 1) / 2 * log_n1
        + (L - 1) / 2 * log_n2
        + (K * L + 1) / 2 * log_n1n2
    )
    if missing_model == "mnar":
        pen += log_n1n2            # = 2 * (1/2 log n1) + 2 * (1/2 log n2)
    elif missing_model == "mar":
        pen += 0.5 * log_n1n2      # = 1/2 log n1 + 1/2 log n2
    else:
        raise ValueError("missing_model deve essere 'mnar' o 'mar'")
    return pen


# ----------------------------------------------------------------------
# Estrazione parametri
# ----------------------------------------------------------------------
def _params(model, n1, n2, K, L):
    with torch.no_grad():
        p = reparametrized_expanded_params(
            torch.cat((model.variationnal_params, model.model_params)),
            n1, n2, K, L, model.device,
        )
    return tuple(x.detach() for x in p)


# ----------------------------------------------------------------------
# ICL "variazionale" -- quello prescritto dal paper
# ----------------------------------------------------------------------
def compute_icl(model, n1=None, n2=None, K=None, L=None,
                missing_model="mnar", return_parts=False):
    """
    ICL(K, L) = [ J(gamma_hat, theta_hat) - H(q_gamma_hat) ] - penalita'.

    Valori piu' grandi = modello migliore.

    Da chiamare su un modello gia' addestrato (dopo `train_with_LBFGS`).
    """
    n1 = model.n1 if n1 is None else n1
    n2 = model.n2 if n2 is None else n2
    K = model.nq if K is None else K
    L = model.nl if L is None else L

    (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
     tau_1, tau_2, mu_un, s2_a, s2_b, s2_p, s2_q,
     alpha_1, alpha_2, pi) = _params(model, n1, n2, K, L)

    with torch.no_grad():
        # criteria() = -J  (L-BFGS minimizza)
        J = -float(model(no_grad=True).item())
        H = float(
            model.entropy_rx(rho_a, rho_b, rho_p, rho_q, tau_1, tau_2).item()
        )

    completed_ll = J - H                      # E_q[log p(X^o, Y, Z, A, B, C, D)]
    pen = icl_penalty(n1, n2, K, L, missing_model)
    icl = completed_ll - pen

    if return_parts:
        return icl, {"elbo": J, "entropy": H,
                     "completed_loglik": completed_ll, "penalty": pen}
    return icl


# ----------------------------------------------------------------------
# Variante "plug-in MAP" (facoltativa, NON e' la formula del paper)
# ----------------------------------------------------------------------
def compute_icl_map(model, n1=None, n2=None, K=None, L=None,
                    missing_model="mnar"):
    """
    Approssimazione alternativa: log-verosimiglianza completa valutata nelle
    assegnazioni MAP (Z, W hard) e nelle medie a posteriori nu di A, B, C, D,
    con alpha e sigma^2 *profilati* in quei valori (per rispettare il "max su
    theta" di Proposition 1). pi e mu restano quelli stimati dal VEM.

    Differisce sistematicamente da `compute_icl` (in genere e' piu' grande,
    perche' ignora l'incertezza di q) e NON e' confrontabile con i valori del
    paper: usarla solo come diagnostica.
    """
    n1 = model.n1 if n1 is None else n1
    n2 = model.n2 if n2 is None else n2
    K = model.nq if K is None else K
    L = model.nl if L is None else L

    (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
     tau_1, tau_2, mu_un, s2_a, s2_b, s2_p, s2_q,
     alpha_1, alpha_2, pi) = _params(model, n1, n2, K, L)

    device = model.device

    # --- assegnazioni MAP ---
    k_hat = tau_1.argmax(dim=1)                       # (n1,)
    l_hat = tau_2.argmax(dim=1)                       # (n2,)

    # --- log p(Y) e log p(Z) con alpha profilato: alpha_k = n_k / n1 ---
    n_k = torch.bincount(k_hat, minlength=K).double()
    m_l = torch.bincount(l_hat, minlength=L).double()
    nz_k, nz_l = n_k[n_k > 0], m_l[m_l > 0]
    log_pY = (nz_k * torch.log(nz_k / n1)).sum()
    log_pZ = (nz_l * torch.log(nz_l / n2)).sum()

    # --- log p(A), p(B), p(C), p(D) con sigma^2 profilato ---
    def _log_gauss_profiled(x):
        n = x.numel()
        s2_hat = (x.flatten() ** 2).sum() / n
        return -n / 2 * (torch.log(torch.tensor(2 * np.pi, dtype=torch.double,
                                                device=device))
                         + 1 + torch.log(s2_hat))

    log_pA = _log_gauss_profiled(nu_a)
    log_pB = _log_gauss_profiled(nu_b)
    log_pC = _log_gauss_profiled(nu_p)
    log_pD = _log_gauss_profiled(nu_q)

    # --- log p(X^(o) | Y, Z, A, B, C, D) ---
    mu = mu_un.squeeze()
    A = nu_a.reshape(n1, 1)
    B = nu_b.reshape(n1, 1)
    C = nu_p.reshape(1, n2)
    D = nu_q.reshape(1, n2)

    sig_1 = torch.sigmoid(mu + A + B + C + D)         # X_ij = 1
    sig_0 = torch.sigmoid(mu + A - B + C - D)         # X_ij = 0
    pi_ij = pi[k_hat[:, None], l_hat[None, :]]

    eps = 1e-12
    p_1 = (pi_ij * sig_1).clamp(min=eps)
    p_0 = ((1 - pi_ij) * sig_0).clamp(min=eps)
    p_NA = (1 - pi_ij * sig_1 - (1 - pi_ij) * sig_0).clamp(min=eps)

    # NB nella repo: votes == 1 -> voto favorevole (X=1)
    #                votes == -1 -> voto contrario (X=0)
    #                votes == 0 -> dato MANCANTE (NA)
    idx_p = torch.as_tensor(model.indices_p, device=device, dtype=torch.long)
    idx_n = torch.as_tensor(model.indices_n, device=device, dtype=torch.long)
    idx_z = torch.as_tensor(model.indices_zeros, device=device, dtype=torch.long)

    log_pX = (
        torch.log(p_1[idx_p[:, 0], idx_p[:, 1]]).sum()
        + torch.log(p_0[idx_n[:, 0], idx_n[:, 1]]).sum()
        + torch.log(p_NA[idx_z[:, 0], idx_z[:, 1]]).sum()
    )

    log_joint = log_pY + log_pZ + log_pA + log_pB + log_pC + log_pD + log_pX
    return float(log_joint.item()) - icl_penalty(n1, n2, K, L, missing_model)
