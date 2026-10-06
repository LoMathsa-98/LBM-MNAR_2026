# icl.py
"""
Criterio ICL asintotico per LBM-MNAR (Frisch, Leger, Grandvalet 2022),
generalizzato a modelli vincolati tramite i flag use_A/B/C/D del modello.

    ICL = E_q[log p(X^o, Y, Z, A, B, C, D)] - penalita'
        = J - H(q) - penalita'

dove J e' l'ELBO e H l'entropia di q (entrambe flag-aware in LBM_NMAR).

Penalita' generale (a,b,c,d = 1 se il blocco A,B,C,D e' presente):

    0.5 * [ (K-1+a+b) log n1 + (L-1+c+d) log n2 + (K*L+1) log(n1 n2) ]

    mnar (1,1,1,1) -> penalita' originale del paper
    mar  (1,0,1,0) -> versione MAR, Eqs. (20)-(21)
    mcar (0,0,0,0) -> solo K, L, pi, mu

ATTENZIONE: model(...) / criteria() restituisce -J (L-BFGS minimizza).
"""

import numpy as np
import torch

from utils import reparametrized_expanded_params


def icl_penalty(n1, n2, K, L, a=1, b=1, c=1, d=1):
    """Penalita' BIC-like; a,b,c,d in {0,1} indicano i blocchi presenti."""
    a, b, c, d = int(a), int(b), int(c), int(d)
    return 0.5 * ((K - 1 + a + b) * np.log(n1)
                  + (L - 1 + c + d) * np.log(n2)
                  + (K * L + 1) * np.log(n1 * n2))


def _params(model, n1, n2, K, L):
    with torch.no_grad():
        p = reparametrized_expanded_params(
            torch.cat((model.variationnal_params, model.model_params)),
            n1, n2, K, L, model.device,
        )
    return tuple(x.detach() for x in p)


def compute_icl(model, n1=None, n2=None, K=None, L=None,
                return_parts=False, with_diagnostics=False):
    """
    ICL(K, L) = [J - H(q)] - penalita'(flag del modello).

    Valori piu' grandi = modello migliore. Da chiamare su un modello addestrato.

    parts (se return_parts=True):
        elbo, entropy, completed_loglik, penalty, flags,
        valid  : True se J e' finito e J < 0 (J <= log p(X^o) <= 0 perche'
                 X^o e' discreta). Se False (J = +inf dal ramo 'expectation<0'
                 del termine NA, oppure NaN da classe vuota) l'ICL NON va usato.
        icl_positive : True se l'ICL e' > 0 (sospetto: log p(blocco) e' una
                 densita' e puo' diventare grande se sigma^2 -> 0).
        diagnostics (solo con with_diagnostics=True): model.diagnostics().
    """
    n1 = model.n1 if n1 is None else n1
    n2 = model.n2 if n2 is None else n2
    K = model.nq if K is None else K
    L = model.nl if L is None else L

    (nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
     tau_1, tau_2, *_rest) = _params(model, n1, n2, K, L)

    with torch.no_grad():
        J = -float(model(no_grad=True).item())          # criteria() = -J
        H = float(model.entropy_rx(
            rho_a, rho_b, rho_p, rho_q, tau_1, tau_2).item())

    completed_ll = J - H
    flags = (int(model.use_A), int(model.use_B),
             int(model.use_C), int(model.use_D))
    pen = icl_penalty(n1, n2, K, L, *flags)
    icl = completed_ll - pen

    if return_parts:
        parts = {
            "elbo": J, "entropy": H, "completed_loglik": completed_ll,
            "penalty": pen, "flags": flags,
            "valid": bool(np.isfinite(J) and J < 0),
            "icl_positive": bool(np.isfinite(icl) and icl > 0),
        }
        if with_diagnostics:
            try:
                parts["diagnostics"] = model.diagnostics()
            except Exception as e:                      # diagnostica opzionale
                parts["diagnostics"] = {"error": str(e)[:200]}
        return icl, parts
    return icl

