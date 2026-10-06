import torch
import numpy as np
from torch import nn
from utils import reparametrized_expanded_params, d2_DL3_XO

BLOCKS = ("A", "B", "C", "D")


class LBM_NMAR(nn.Module):
    """
    LBM con missingness MNAR (Frisch et al. 2022), generalizzato a modelli vincolati.

    use_A / use_B / use_C / use_D  indicano quali blocchi gaussiani di propensione
    sono PRESENTI nel modello:
        A_i (riga, MAR), B_i (riga, MNAR), C_j (colonna, MAR), D_j (colonna, MNAR).
    Un blocco con flag False e' *assente* (la variabile e' identicamente 0): non
    entra in entropia, E[log p(blocco)], Taylor e penalita' ICL. I suoi parametri
    restano nel vettore (struttura invariata) ma hanno gradiente esattamente 0 e
    non influenzano in alcun modo il criterio.

        mnar : A B C D      (tutti True, comportamento identico all'originale)
        noB  : A   C D
        noAB :     C D
        mar  : A   C
        mcar : (nessuno)    (resta solo mu)
    """

    def __init__(self, init_parameters, votes, shapes, device, device2=None,
                 use_A=True, use_B=True, use_C=True, use_D=True):
        super().__init__()
        self.indices_p = np.argwhere(votes == 1)
        self.indices_n = np.argwhere(votes == -1)
        self.indices_zeros = np.argwhere(votes == 0)

        self.n1 = shapes[0]
        self.n2 = shapes[1]
        self.nq = shapes[2]
        self.nl = shapes[3]
        self.device = device
        self.device2 = device2
        self.use_A = bool(use_A)
        self.use_B = bool(use_B)
        self.use_C = bool(use_C)
        self.use_D = bool(use_D)
        self._mask_cache = None
        lengamma = (
            4 * self.n1
            + 4 * self.n2
            + (self.n1 * (self.nq - 1))
            + (self.n2 * (self.nl - 1))
        )
        self.variationnal_params = nn.Parameter(
            init_parameters[:lengamma].clone()
        )
        self.model_params = nn.Parameter(init_parameters[lengamma:].clone())

    # ------------------------------------------------------------------
    # utilita'
    # ------------------------------------------------------------------
    @property
    def flags(self):
        return {"use_A": self.use_A, "use_B": self.use_B,
                "use_C": self.use_C, "use_D": self.use_D}

    def raw_block_slices(self):
        """Indici (nel vettore completo gamma+theta) dei parametri grezzi di
        ciascun blocco: nu, rho (variazionali) e sigma^2 (modello)."""
        n1, n2 = self.n1, self.n2
        n_gamma = (4 * n1 + 4 * n2 + n1 * (self.nq - 1) + n2 * (self.nl - 1))
        r = np.arange
        return {
            "A": {"nu": r(0, n1), "rho": r(n1, 2 * n1),
                  "sigma2": r(n_gamma + 1, n_gamma + 2)},
            "B": {"nu": r(2 * n1, 3 * n1), "rho": r(3 * n1, 4 * n1),
                  "sigma2": r(n_gamma + 2, n_gamma + 3)},
            "C": {"nu": r(4 * n1, 4 * n1 + n2),
                  "rho": r(4 * n1 + n2, 4 * n1 + 2 * n2),
                  "sigma2": r(n_gamma + 3, n_gamma + 4)},
            "D": {"nu": r(4 * n1 + 2 * n2, 4 * n1 + 3 * n2),
                  "rho": r(4 * n1 + 3 * n2, 4 * n1 + 4 * n2),
                  "sigma2": r(n_gamma + 4, n_gamma + 5)},
        }

    def _unpack(self):
        return reparametrized_expanded_params(
            torch.cat((self.variationnal_params, self.model_params)),
            self.n1, self.n2, self.nq, self.nl, self.device,
        )

    def _effective(self, nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q):
        """(nu, rho) effettivi: blocchi assenti -> zeri esatti (nessun grafo)."""
        zr = torch.zeros_like(nu_a)
        zc = torch.zeros_like(nu_p)
        return (
            (nu_a, rho_a) if self.use_A else (zr, zr),
            (nu_b, rho_b) if self.use_B else (zr, zr),
            (nu_p, rho_p) if self.use_C else (zc, zc),
            (nu_q, rho_q) if self.use_D else (zc, zc),
        )

    def _masks(self):
        if self._mask_cache is None:
            def m(idx):
                z = torch.zeros((self.n1, self.n2), dtype=torch.double,
                                device=self.device)
                if len(idx):
                    z[torch.as_tensor(idx[:, 0], dtype=torch.long),
                      torch.as_tensor(idx[:, 1], dtype=torch.long)] = 1.0
                return z
            self._mask_cache = (m(self.indices_p), m(self.indices_n),
                                m(self.indices_zeros))
        return self._mask_cache

    # ------------------------------------------------------------------
    # criterio (-J)
    # ------------------------------------------------------------------
    def forward(self, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.criteria()
        else:
            return self.criteria()

    def _terms(self):
        (
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
            tau_1, tau_2, mu_un,
            sigma_sq_a, sigma_sq_b, sigma_sq_p, sigma_sq_q,
            alpha_1, alpha_2, pi,
        ) = self._unpack()
        if torch.any(tau_1.sum(dim=0) < 0.5):
            print("One empty row class, algo stoped")
            return None
        if torch.any(tau_2.sum(dim=0) < 0.5):
            print("One empty column class, algo stoped")
            return None

        t = {"H": self.entropy_rx(rho_a, rho_b, rho_p, rho_q, tau_1, tau_2)}
        if self.use_A:
            t["ll_A"] = self.expectation_loglike_A(nu_a, rho_a, sigma_sq_a)[0]
        if self.use_B:
            t["ll_B"] = self.expectation_loglike_B(nu_b, rho_b, sigma_sq_b)[0]
        if self.use_C:
            t["ll_C"] = self.expectation_loglike_P(nu_p, rho_p, sigma_sq_p)[0]
        if self.use_D:
            t["ll_D"] = self.expectation_loglike_Q(nu_q, rho_q, sigma_sq_q)[0]
        t["ll_Y"] = self.expectation_loglike_Y1(tau_1, alpha_1)
        t["ll_Z"] = self.expectation_loglike_Y2(tau_2, alpha_2)

        (na, ra), (nb, rb), (npp, rp), (nqq, rq) = self._effective(
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q)
        t["ll_X"] = self.expectation_loglike_X_cond_ABPY1Y2(
            mu_un, na, nb, npp, nqq, ra, rb, rp, rq, tau_1, tau_2, pi)

        total = 0
        for v in t.values():
            total = total + v
        t["total"] = total
        return t

    def criteria(self):
        t = self._terms()
        if t is None:
            return torch.tensor(np.nan, device=self.device)
        return -t["total"]

    def entropy_rx(self, rho_a, rho_b, rho_p, rho_q, tau_1, tau_2):
        n1, n2 = self.n1, self.n2
        n_gauss = (int(self.use_A) + int(self.use_B)) * n1 \
            + (int(self.use_C) + int(self.use_D)) * n2
        c = torch.log(
            torch.tensor(2 * np.pi, dtype=torch.double, device=self.device)
        ) + 1
        H = (
            0.5 * n_gauss * c
            - torch.sum(tau_1 * torch.log(tau_1))
            - torch.sum(tau_2 * torch.log(tau_2))
        )
        if self.use_A:
            H = H + 0.5 * torch.sum(torch.log(rho_a))
        if self.use_B:
            H = H + 0.5 * torch.sum(torch.log(rho_b))
        if self.use_C:
            H = H + 0.5 * torch.sum(torch.log(rho_p))
        if self.use_D:
            H = H + 0.5 * torch.sum(torch.log(rho_q))
        return H

    # ------------------------------------------------------------------
    # diagnostica
    # ------------------------------------------------------------------
    @torch.no_grad()
    def diagnostics(self):
        """Scomposizione di J = H + E_q[log p] nei singoli termini e statistiche
        per blocco (solo blocchi attivi; per i blocchi assenti: NaN)."""
        (
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
            tau_1, tau_2, mu_un,
            s2_a, s2_b, s2_p, s2_q, _al1, _al2, _pi,
        ) = self._unpack()
        t = self._terms()
        if t is None:
            return None
        out = {k: float(v.item()) for k, v in t.items()}
        out["elbo"] = out["total"]
        h_tau = float(-(tau_1 * torch.log(tau_1)).sum()
                      - (tau_2 * torch.log(tau_2)).sum())
        out["H_tau"] = h_tau
        out["H_gauss"] = out["H"] - h_tau
        out["ll_discrete"] = out["ll_Y"] + out["ll_Z"] + out["ll_X"]
        out["ll_gauss_prior"] = sum(out.get("ll_" + b, 0.0) for b in BLOCKS)
        out["completed_ll"] = out["elbo"] - out["H"]

        stats = {"A": (nu_a, rho_a, s2_a), "B": (nu_b, rho_b, s2_b),
                 "C": (nu_p, rho_p, s2_p), "D": (nu_q, rho_q, s2_q)}
        nan = float("nan")
        for b, (nu, rho, s2) in stats.items():
            if getattr(self, "use_" + b):
                out["sigma2_" + b] = float(s2.item())
                out["nu_rms_" + b] = float(torch.sqrt((nu ** 2).mean()))
                out["nu_absmax_" + b] = float(nu.abs().max())
                out["rho_mean_" + b] = float(rho.mean())
                out["rho_min_" + b] = float(rho.min())
                out["rho_clamp_frac_" + b] = float(
                    (rho <= 1.0001e-6).double().mean())
            else:
                for k in ("sigma2_", "nu_rms_", "nu_absmax_", "rho_mean_",
                          "rho_min_", "rho_clamp_frac_"):
                    out[k + b] = nan

        (na, _), (nb, _), (npp, _), (nqq, _) = self._effective(
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q)
        eta_p = mu_un + na + nb + npp + nqq
        eta_m = mu_un + na - nb + npp - nqq
        out["eta_abs_max"] = float(
            torch.maximum(eta_p.abs().max(), eta_m.abs().max()))
        return out

    @torch.no_grad()
    def mc_check_X(self, n_draws=10, seed=0):
        """Confronta il termine E_q[log p(X^(o) | ...)] calcolato con Taylor del
        secondo ordine con una stima Monte Carlo (campionando A,B,C,D da q e
        sommando esattamente su (k,l) con pesi tau)."""
        (
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q,
            tau_1, tau_2, mu_un, _s1, _s2, _s3, _s4, _al1, _al2, pi,
        ) = self._unpack()
        t = self._terms()
        if t is None:
            return None
        taylor = float(t["ll_X"].item())
        (na, ra), (nb, rb), (npp, rp), (nqq, rq) = self._effective(
            nu_a, rho_a, nu_b, rho_b, nu_p, rho_p, nu_q, rho_q)
        K, L = self.nq, self.nl
        mp, mn, mz = self._masks()
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        dt = na.dtype

        def draw(nu, rho):
            return nu + torch.sqrt(rho) * torch.randn(
                nu.shape, generator=g, device=self.device, dtype=dt)

        logpi = torch.log(pi)
        log1mpi = torch.log1p(-pi)
        T_p = tau_1 @ logpi @ tau_2.T
        T_n = tau_1 @ log1mpi @ tau_2.T
        vals = []
        for _ in range(int(n_draws)):
            a, b = draw(na, ra), draw(nb, rb)
            c, d = draw(npp, rp), draw(nqq, rq)
            eta_p = mu_un + a + b + c + d
            eta_m = mu_un + a - b + c - d
            logS1 = torch.nn.functional.logsigmoid(eta_p)
            logS0 = torch.nn.functional.logsigmoid(eta_m)
            S1, S0 = torch.sigmoid(eta_p), torch.sigmoid(eta_m)
            v = ((T_p + logS1) * mp).sum() + ((T_n + logS0) * mn).sum()
            acc = torch.zeros_like(S1)
            for k in range(K):
                for l in range(L):
                    w = tau_1[:, k:k + 1] * tau_2[:, l].reshape(1, -1)
                    acc = acc + w * torch.log(
                        (1 - pi[k, l] * S1 - (1 - pi[k, l]) * S0).clamp(min=1e-300))
            v = v + (acc * mz).sum()
            vals.append(float(v.item()))
        vals = np.asarray(vals)
        mc = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        return {"taylor": taylor, "mc": mc, "mc_se": se,
                "abs_diff": abs(taylor - mc),
                "rel_diff": abs(taylor - mc) / max(abs(mc), 1e-12)}

    # ------------------------------------------------------------------
    # termini di J (invariati rispetto all'originale)
    # ------------------------------------------------------------------
    def expectation_loglike_A(self, nu_a, rho_a, sigma_sq_a):
        n1 = self.n1
        return -n1 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=self.device)
            )
            + torch.log(sigma_sq_a)
        ) - 1 / (2 * sigma_sq_a) * torch.sum(rho_a + nu_a ** 2)

    def expectation_loglike_B(self, nu_b, rho_b, sigma_sq_b):
        n1 = self.n1
        return -n1 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=self.device)
            )
            + torch.log(sigma_sq_b)
        ) - 1 / (2 * sigma_sq_b) * torch.sum(rho_b + nu_b ** 2)

    def expectation_loglike_P(self, nu_p, rho_p, sigma_sq_p):
        n2 = self.n2
        return -n2 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=self.device)
            )
            + torch.log(sigma_sq_p)
        ) - 1 / (2 * sigma_sq_p) * torch.sum(rho_p + nu_p ** 2)

    def expectation_loglike_Q(self, nu_q, rho_q, sigma_sq_q):
        n2 = self.n2
        return -n2 / 2 * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=self.device)
            )
            + torch.log(sigma_sq_q)
        ) - 1 / (2 * sigma_sq_q) * torch.sum(rho_q + nu_q ** 2)

    def expectation_loglike_Y1(self, tau_1, alpha_1):
        n1, nq = self.n1, self.nq
        return tau_1.sum(0) @ torch.log(alpha_1)

    def expectation_loglike_Y2(self, tau_2, alpha_2):
        n2, nl = self.n2, self.nl
        return tau_2.sum(0) @ torch.log(alpha_2).t()

    def expectation_loglike_X_cond_ABPY1Y2(
        self,
        mu_un,
        nu_a,
        nu_b,
        nu_p,
        nu_q,
        rho_a,
        rho_b,
        rho_p,
        rho_q,
        tau_1,
        tau_2,
        pi,
    ):
        n1, n2, nq, nl = self.n1, self.n2, self.nq, self.nl
        indices_p, indices_n, indices_zeros = (
            self.indices_p,
            self.indices_n,
            self.indices_zeros,
        )
        i_p_one = indices_p[:, 0]
        i_m_one = indices_n[:, 0]
        i_zeros = indices_zeros[:, 0]
        j_p_one = indices_p[:, 1]
        j_m_one = indices_n[:, 1]
        j_zeros = indices_zeros[:, 1]

        ## POSITIVES ###
        xp = nu_a[i_p_one].flatten() + nu_p[:, j_p_one].flatten()
        yp = nu_b[i_p_one].flatten() + nu_q[:, j_p_one].flatten()
        sig_p = torch.sigmoid(
            mu_un
            + nu_a[i_p_one]
            + nu_b[i_p_one]
            + nu_p[:, j_p_one].t()
            + nu_q[:, j_p_one].t()
        )
        der2_sig_p = (-sig_p * (1 - sig_p)).flatten()
        sum_var_p = (
            rho_a[i_p_one].flatten()
            + rho_p[:, j_p_one].flatten()
            + rho_b[i_p_one].flatten()
            + rho_q[:, j_p_one].flatten()
        )

        f = lambda x, y: torch.log(
            pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1)
        )
        expectation_taylor_p = (
            tau_1[i_p_one].view(-1, nq, 1)
            * tau_2[j_p_one].view(-1, 1, nl)
            * (f(xp, yp) + 0.5 * (der2_sig_p * sum_var_p).view(-1, 1, 1))
        ).sum()

        ### NEGATIVES ###
        xn = nu_a[i_m_one].flatten() + nu_p[:, j_m_one].flatten()
        yn = nu_b[i_m_one].flatten() + nu_q[:, j_m_one].flatten()
        sig_m = torch.sigmoid(
            mu_un
            + nu_a[i_m_one]
            - nu_b[i_m_one]
            + nu_p[:, j_m_one].t()
            - nu_q[:, j_m_one].t()
        )
        der2_sig_m = -(sig_m * (1 - sig_m)).flatten()
        sum_var_m = (
            rho_a[i_m_one].flatten()
            + rho_p[:, j_m_one].flatten()
            + rho_b[i_m_one].flatten()
            + rho_q[:, j_m_one].flatten()
        )
        f = lambda x, y: torch.log(
            (1 - pi).view(1, nq, nl)
            * torch.sigmoid(mu_un + x - y).view(-1, 1, 1)
        )

        expectation_taylor_m = (
            tau_1[i_m_one].view(-1, nq, 1)
            * tau_2[j_m_one].view(-1, 1, nl)
            * (f(xn, yn) + 0.5 * (der2_sig_m * sum_var_m).view(-1, 1, 1))
        ).sum()

        ### ZEROS ###

        f = lambda x, y: torch.log(
            1
            - pi.view(1, nq, nl) * torch.sigmoid(mu_un + x + y).view(-1, 1, 1)
            - (1 - pi.view(1, nq, nl))
            * torch.sigmoid(mu_un + x - y).view(-1, 1, 1)
        )
        xz = nu_a[i_zeros].flatten() + nu_p[:, j_zeros].flatten()
        yz = nu_b[i_zeros].flatten() + nu_q[:, j_zeros].flatten()

        if self.device2:
            der_x = d2_DL3_XO(
                xz.view(-1, 1, 1).to(self.device2),
                yz.view(-1, 1, 1).to(self.device2),
                mu_un.to(self.device2),
                pi.view(1, nq, nl).to(self.device2),
                "x",
            ).to(self.device)
        else:
            der_x = d2_DL3_XO(
                xz.view(-1, 1, 1),
                yz.view(-1, 1, 1),
                mu_un,
                pi.view(1, nq, nl),
                "x",
            )
        der_y = d2_DL3_XO(
            xz.view(-1, 1, 1),
            yz.view(-1, 1, 1),
            mu_un,
            pi.view(1, nq, nl),
            "y",
        )
        tau_12_ij = tau_1[i_zeros].view(-1, nq, 1) * tau_2[j_zeros].view(
            -1, 1, nl
        )
        expectation_taylor_zeros = (
            tau_12_ij
            * (
                f(xz, yz)
                + 0.5
                * (
                    der_x
                    * (
                        rho_a[i_zeros].flatten() + rho_p[:, j_zeros].flatten()
                    ).view(-1, 1, 1)
                    + der_y
                    * (
                        rho_b[i_zeros].flatten() + rho_q[:, j_zeros].flatten()
                    ).view(-1, 1, 1)
                )
            )
        ).sum()

        expectation = (
            expectation_taylor_p
            + expectation_taylor_m
            + expectation_taylor_zeros
        )
        return (
            expectation
            if expectation < 0
            else torch.tensor(np.inf, device=self.device)
        )

