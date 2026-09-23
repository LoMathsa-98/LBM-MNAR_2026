from lbfgs import FullBatchLBFGS
import numpy as np
import torch

def train_with_LBFGS(
    model,
    loglike_dist_tol=1e-4,
    max_iter_EM=50000,
    max_iter_LBFGS=300,
    norm_grad_tol=1e-4,
    initial_learning_rate=1.0,
    hessian_history_size=100,
    early_stopping=False,
    loglike_diff_breaking_cond=1e-3,
    divide_by_line_search=2,
):
    try:
        print("-" * 80, "\nStart training LBM MNAR", "\n", "-" * 80)
        print("Number of row classes : ", model.nq)
        print("Number of col classes : ", model.nl)
        print(
            f""" EM step  |   LBFGS iter  | criteria |"""
        )
        eobj_prec = 0
        success = False
        for i_step in range(0, max_iter_EM):
            line_search = "Armijo"
            optimizer = FullBatchLBFGS(
                [model.variationnal_params]
                if i_step % 2 == 0
                else [model.model_params],
                lr=initial_learning_rate,
                history_size=hessian_history_size,
                line_search=line_search,
                debug=True,
            )
            func_evals = 0
            optimizer.zero_grad()
            obj = model()
            obj.backward()

            if np.abs(
                eobj_prec - obj.item()
            ) < loglike_diff_breaking_cond and (i_step > 1):
                print("Training Finished.")
                success = True
                break
            if early_stopping == i_step + 1:
                print(
                    "Early stopping reached. stopping. Considered as success."
                )
                success = True
                break
            eobj_prec = obj.item()
            grad = optimizer._gather_flat_grad()
            func_evals += 1
            f_old = obj.item()

            for n_iter in range(max_iter_LBFGS):
                # define closure for line search
                def closure():
                    loss_fn = model(no_grad=True)
                    return loss_fn

                # perform line search step
                options = {
                    "closure": closure,
                    "current_loss": obj,
                    "eta": divide_by_line_search,
                    "max_ls": 150,
                    "interpolate": False,
                    "inplace": True,
                    "ls_debug": False,
                    "damping": False,
                    "eps": 1e-2,
                    "c1": 0.5,
                    "c2": 0.95,
                }

                obj, lr, backtracks, clos_evals, desc_dir, fail = optimizer.step(
                    options=options
                )
                optimizer.zero_grad()
                obj = model()
                obj.backward()
                grad = optimizer._gather_flat_grad()

                if optimizer.state["global_state"]["fail_skips"] > 0:
                    raise Exception("BFGS failed : fail_skip")
                if obj.item() < 0:
                    raise Exception("BFGS failed :  obj inf or <0")
                if np.isnan(obj.item()):
                    raise Exception(
                        "Objective function is NAN. Probably due to empty class"
                    )
                print(
                    f""" {i_step}  |   {n_iter + 1}  | {obj.item():.5f} |"""
                )
                if (
                    torch.norm(grad) < norm_grad_tol
                    or np.abs(obj.item() - f_old) < loglike_dist_tol
                ):
                    print("-" * 30, " Optimizing next EM step ", "-" * 30)
                    print(
                        f""" EM step  |   LBFGS iter  | criteria |"""
                    )
                    break
                f_old = obj.item()

        return (success, model().item())
    except Exception as e:
        print(e)
        return (False, obj.item())
        
        


import time
from collections import deque


def train_with_LBFGS_adaptive(
    model,
    time_budget_s=700,
    loglike_diff_breaking_cond=1e-3,
    tol_growth=10.0,
    osc_window=20,
    osc_band=5e-3,
    osc_sign_frac=0.6,
    plateau_std=1e-4,
    plateau_range=1e-4,
    max_iter_EM=50000,
    max_iter_LBFGS=300,
    norm_grad_tol=1e-4,
    loglike_dist_tol=1e-4,
    initial_learning_rate=1.0,
    hessian_history_size=100,
    divide_by_line_search=2,
):
    """Drop-in replacement di train_with_LBFGS con stop adattivo.

    Aggiunge tre meccanismi di stop:
      1) Tolleranza adattiva sul tempo (start strict, end loose).
      2) Rilevatore di oscillazione (range piccolo + molti cambi di segno).
      3) Rilevatore di plateau (valori quasi costanti).
    """
    t0 = time.time()
    hist = deque(maxlen=osc_window)
    obj = None
    try:
        print("-" * 80, "\nStart training LBM MNAR (adaptive)", "\n", "-" * 80)
        print("Number of row classes : ", model.nq)
        print("Number of col classes : ", model.nl)
        eobj_prec = 0
        success = False
        for i_step in range(max_iter_EM):
            optimizer = FullBatchLBFGS(
                [model.variationnal_params] if i_step % 2 == 0
                else [model.model_params],
                lr=initial_learning_rate,
                history_size=hessian_history_size,
                line_search="Armijo",
                debug=False,
            )
            optimizer.zero_grad()
            obj = model()
            obj.backward()
            f_old = obj.item()

            # --- tolleranza adattiva sul tempo ---
            frac = min((time.time() - t0) / time_budget_s, 1.0)
            tol_now = loglike_diff_breaking_cond * (1 + (tol_growth - 1) * frac ** 2)
            if i_step > 1 and abs(eobj_prec - obj.item()) < tol_now:
                print(f"[stop EM: tol={tol_now:.2e}, frac={frac:.2f}]")
                success = True
                break
            eobj_prec = obj.item()

            for n_iter in range(max_iter_LBFGS):
                def closure():
                    return model(no_grad=True)

                options = {
                    "closure": closure,
                    "current_loss": obj,
                    "eta": divide_by_line_search,
                    "max_ls": 150,
                    "interpolate": False,
                    "inplace": True,
                    "ls_debug": False,
                    "damping": False,
                    "eps": 1e-2,
                    "c1": 0.5,
                    "c2": 0.95,
                }
                obj, lr, backtracks, clos_evals, desc_dir, fail = optimizer.step(options=options)
                optimizer.zero_grad()
                obj = model()
                obj.backward()
                grad = optimizer._gather_flat_grad()

                if optimizer.state["global_state"]["fail_skips"] > 0:
                    raise Exception("BFGS failed : fail_skip")
                if obj.item() < 0:
                    raise Exception("BFGS failed : obj inf or <0")
                if np.isnan(obj.item()):
                    raise Exception("Objective is NAN (empty class?)")

                if torch.norm(grad) < norm_grad_tol or abs(obj.item() - f_old) < loglike_dist_tol:
                    break
                f_old = obj.item()

            # --- rilevatore di oscillazione / plateau ---
            hist.append(obj.item())
            if len(hist) == osc_window:
                vals = np.array(hist)
                rng = vals.max() - vals.min()
                diffs = np.diff(vals)
                sign_flips = int(np.sum(np.abs(np.diff(np.sign(diffs))) / 2))

                if rng < osc_band and sign_flips > osc_sign_frac * (osc_window - 2):
                    print(f"[stop: oscillazione, rng={rng:.2e}, flips={sign_flips}]")
                    success = True
                    break

                if vals.std() < plateau_std and (vals[-1] - vals.min()) < plateau_range:
                    print(f"[stop: plateau, std={vals.std():.2e}]")
                    success = True
                    break

        return (success, obj.item() if obj is not None else float('nan'))
    except Exception as e:
        print(e)
        return (False, obj.item() if obj is not None else float('nan'))
