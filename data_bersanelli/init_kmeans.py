import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logit
from utils import inv_softplus, shrink_simplex

def init_from_kmeans(n1, n2, nq, nl, X_int, epsilon=0.05):
    """
    Inizializzazione di init_random_params basata su KMeans.
    Sostituisce tau_1 e tau_2 casuali con assegnazioni KMeans.
    Il resto dei parametri rimane come nell'originale.
    """
    # imputa i missing con 0.5 per KMeans
    X_float = X_int.astype(float)
    X_float[X_float == -1] = 0.5

    # cluster righe (pazienti)
    row_labels = KMeans(n_clusters=nq, n_init=10, random_state=42).fit_predict(X_float)
    # cluster colonne (geni)
    col_labels = KMeans(n_clusters=nl, n_init=10, random_state=42).fit_predict(X_float.T)

    # converti label in soft assignments one-hot con epsilon smoothing
    def labels_to_tau(labels, n_clusters):
        n = len(labels)
        tau = np.full((n, n_clusters), epsilon / n_clusters)
        for i, c in enumerate(labels):
            tau[i, c] = 1.0 - epsilon + epsilon / n_clusters
        return tau

    tau_1 = labels_to_tau(row_labels, nq)   # (n1, nq)
    tau_2 = labels_to_tau(col_labels, nl)   # (n2, nl)

    # parametri modello: stessi default di init_random_params
    mu_un       = np.random.uniform(-4.5, -3.5)
    sigma_sq_a  = np.random.uniform(0.4, 0.7)
    sigma_sq_b  = np.random.uniform(0.4, 0.7)
    sigma_sq_p  = np.random.uniform(0.4, 0.7)
    sigma_sq_q  = np.random.uniform(0.4, 0.7)
    alpha_1     = (np.ones(nq) / nq).reshape((nq, 1))
    alpha_2     = (np.ones(nl) / nl).reshape((1, nl))
    pi          = np.random.uniform(0.2, 0.8, (nq, nl))
    nu_a        = np.random.uniform(-0.5, 0.5, (n1, 1))
    nu_b        = np.random.uniform(-0.5, 0.5, (n1, 1))
    nu_p        = np.random.uniform(-0.5, 0.5, (1, n2))
    nu_q        = np.random.uniform(-0.5, 0.5, (1, n2))
    rho_a       = 1e-5 * np.ones((n1, 1))
    rho_b       = 1e-5 * np.ones((n1, 1))
    rho_p       = 1e-5 * np.ones((1, n2))
    rho_q       = 1e-5 * np.ones((1, n2))

    # costruisci theta (parametri modello) — identico a init_random_params
    theta = np.concatenate((
        (mu_un,),
        (inv_softplus(sigma_sq_a),),
        (inv_softplus(sigma_sq_b),),
        (inv_softplus(sigma_sq_p),),
        (inv_softplus(sigma_sq_q),),
        logit(shrink_simplex(alpha_1.T).flatten()),
        logit(shrink_simplex(alpha_2).flatten()),
        logit(pi.flatten()),
    ))

    # costruisci gamma (parametri variazionali) con tau da KMeans
    gamma = np.concatenate((
        nu_a.flatten(),
        inv_softplus(rho_a.flatten()),
        nu_b.flatten(),
        inv_softplus(rho_b.flatten()),
        nu_p.flatten(),
        inv_softplus(rho_p.flatten()),
        nu_q.flatten(),
        inv_softplus(rho_q.flatten()),
        logit(shrink_simplex(tau_1).flatten()),   # <- KMeans invece di random
        logit(shrink_simplex(tau_2).flatten()),   # <- KMeans invece di random
    ))

    return np.concatenate((gamma, theta))