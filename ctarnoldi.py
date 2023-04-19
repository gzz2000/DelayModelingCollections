# coordinate-transformed arnoldi

import numpy as np
import scipy.linalg as lg
import pdb

# assume single driver at pin 0.
# returns an (n+1)*(n+1) matrix, (0, 0) inserted as real driver
def build_matrix_dr(rc, driver_rd):
    assert rc.endpoints[0] == (0, 'O'), 'We assume root to be #0 node'
    C = np.zeros((rc.n, rc.n), dtype=np.float32)
    G = np.zeros((rc.n, rc.n), dtype=np.float32)
    # connect driver (0) and root (1) with given res
    G[0, 0] = 1. / driver_rd
    for i, c in rc.grounded_caps:
        C[i, i] += c
    for i1, i2, c in rc.coupling_caps:
        C[i1, i1] += c
        C[i1, i2] -= c
        C[i2, i2] += c
        C[i2, i1] -= c
    for i1, i2, r in rc.ress:
        assert r >= 0.00001, 'Too small resistance'
        invr = 1. / r
        G[i1, i1] += invr
        G[i1, i2] -= invr
        G[i2, i2] += invr
        G[i2, i1] -= invr
    for i, c in enumerate(rc.sink_cell_caps):
        C[i + 1, i + 1] += c
    return C, G

# assume single driver at 0
def ctarnoldi(C, G, q, driver_rd):
    n = C.shape[0]
    assert q <= n
    lu, piv = lg.lu_factor(G)
    e = np.zeros(n, dtype=np.float32)
    e[0] = -1. / driver_rd
    u0 = -lg.lu_solve((lu, piv), e)
    z0 = C @ u0
    h00 = np.sqrt(np.dot(u0, z0))
    Hq = np.zeros((q, q), dtype=np.float32)
    zs = [z0 / h00]
    us = [u0 / h00]
    for j in range(1, q + 1):
        w = -lg.lu_solve((lu, piv), zs[j - 1])
        for i in range(max(j - 2, 0), j):
            Hq[i, j - 1] = np.dot(w, zs[i])
            w -= Hq[i, j - 1] * us[i]
        if j >= q: break
        us.append(w)
        zs.append(C @ w)
        hjpj = np.sqrt(np.dot(w, zs[j]))
        Hq[j, j - 1] = hjpj
        if np.abs(hjpj) > 1e-5:
            zs[j] /= hjpj
            us[j] /= hjpj
    Uq = np.stack(us, axis=1)
    return Uq, Hq, lu, piv

def compute_poles_res(Uq, Hq, C, G, Glu, Gpiv, driver_rd):
    eig, eigP = np.linalg.eig(Hq)
    n = Glu.shape[0]
    q = Hq.shape[0]
    e = np.zeros(n, dtype=np.float32)
    e[0] = -1. / driver_rd
    r = lg.lu_solve((Glu, Gpiv), e)
    norm_r = np.sqrt(np.dot(r, C @ r))
    poles = 1. / eig
    residues_mat = norm_r * (Uq[:, :q] @ eigP) * eigP[0, :].reshape(1, q)
    return poles, residues_mat

def exact_solution_compatible_nodriver_nosinkcap(rc, q=4, time_step=0.01, n_steps=5000):
    C, G = build_matrix_dr(rc, 1.)
    # undo sink cap
    for i, c in enumerate(rc.sink_cell_caps):
        C[i + 1, i + 1] -= c
    # clip driver back-current
    C[0, 0] = 0.
    G[0, :] = 0.
    G[0, 0] = 1.
    assert C.shape == (rc.n, rc.n)
    assert G.shape == (rc.n, rc.n)
    Uq, Hq, Glu, Gpiv = ctarnoldi(C, G, q, 1.)
    poles, residues_mat = compute_poles_res(Uq, Hq, C, G, Glu, Gpiv, 1.)
    print(np.sum(residues_mat, axis=1))

    vs = [(0, np.zeros(rc.n))]
    t = 0.
    for i in range(n_steps):
        v = 1. - residues_mat @ np.exp(poles * t)
        vs.append((t, v.reshape(rc.n)))
        t += time_step
    return vs

if __name__ == '__main__':
    from netlist import tree_rc

    C, G = build_matrix_dr(tree_rc, 1.)
    Uq, Hq, Glu, Gpiv = ctarnoldi(C, G, 4, 1.)
    poles, residues_mat = compute_poles_res(Uq, Hq, C, G, Glu, Gpiv, 1.)
    eig, eigP = np.linalg.eig(Hq)

    # # compute exact solution using this reduced-order model
    # vs = exact_solution_compatible_nodriver_nosinkcap(tree_rc)
    # from spice import spice_calc_vt
    # vs_spice = spice_calc_vt(tree_rc, slew=0.,
    #                          time_step=0.01, n_steps=5000,
    #                          method='trapezoidal')
    # import utils
    # utils.plot(tree_rc, [('CT-Arnoldi (Order 4)', vs),
    #                      ('SPICE Trapezoidal', vs_spice)],
    #            title='CT-Arnoldi ROM')
