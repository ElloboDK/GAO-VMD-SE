import numpy as np
from vmdpy import VMD
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist

WIDTH = 3
PROMINENCE = (0.001, None)

def GAO_VMD_SE(data, alpha, K, omega_threshold, multi_score_threshold):
    result = {}
    
    result["O"] = data

    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    result["u"] = u
    result["u_hat"] = u_hat
    result["omega"] = omega

    # wasserstein distance
    result["wasserstein"] = []
    for i in range(K):
        result["wasserstein"].append(wasserstein_distance(data[:-1], u[i, :]))

    X = np.concatenate([[data[:-1]], u],axis=0)
    result["pdist"] = pdist(X,'minkowski',p=2)[:K]

    # multi_score
    result["multi_score"] = []
    u_denoise = []
    u_1, u_2 = [], []
    for i in range(K):
        if np.log10(np.mean(result["omega"][:,i])) > omega_threshold:
            break
        else:
            u_denoise.append(u[i])
            multi_score = np.log10(np.mean(result["omega"][:,i]) * result["pdist"][i])
            result["multi_score"].append(multi_score)
            if multi_score < multi_score_threshold:
                u_1.append(u[i])
            else:
                u_2.append(u[i])
    
    result["u_denoise"] = np.array(u_denoise).sum(axis=0)
    u_1 = np.array(u_1).sum(axis=0)
    u_2 = np.array(u_2).sum(axis=0)
    result["u_re"] = np.array([u_1,u_2])

    return result

