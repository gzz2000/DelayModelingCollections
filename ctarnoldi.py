# coordinate-transformed arnoldi

import numpy as np
import scipy.linalg as lg
import pdb

# assume single driver at pin 0.
# returns an (n+1)*(n+1) matrix, (0, 0) inserted as real driver
def build_matrix_dr(rc, driver_rd):
    assert rc.endpoints[0] == (0, 'O'), 'We assume root to be #0 node'
    C = np.zeros((rc.n + 1, rc.n + 1))
    G = np.zeros((rc.n + 1, rc.n + 1))
    G[0, 0] = 1.
    # connect driver (0) and root (1) with given res
    G[1, 0] = -1. / driver_rd
    G[1, 1] = 1. / driver_rd
    for i, c in rc.grounded_caps:
        C[i + 1, i + 1] += c
    for i1, i2, c in rc.coupling_caps:
        C[i1 + 1, i1 + 1] += c
        C[i1 + 1, i2 + 1] -= c
        C[i2 + 1, i2 + 1] += c
        C[i2 + 1, i1 + 1] -= c
    for i1, i2, r in rc.ress:
        assert r >= 0.00001, 'Too small resistance'
        invr = 1. / r
        G[i1 + 1, i1 + 1] += invr
        G[i1 + 1, i2 + 1] -= invr
        G[i2 + 1, i2 + 1] += invr
        G[i2 + 1, i1 + 1] -= invr
    for i, c in enumerate(rc.sink_cell_caps):
        C[i + 2, i + 2] += c
    return C, G

# assume single driver at 0
def ctarnoldi(C, G, q):
    n = C.shape[0]
    assert q <= n
    lu, piv = lg.lu_factor(G)
    e = np.zeros(n)
    e[0] = -1.
    u0 = -lg.lu_solve((lu, piv), e)
    z0 = C @ u0
    h00 = np.sqrt(np.dot(u0, z0))
    H = np.zeros((q + 1, q))
    z1 = z0 / h00
    u1 = u0 / h00
    zs = [z0, z1]
    us = [u0, u1]
    for j in range(1, q + 1):
        w = -lg.lu_solve((lu, piv), zs[j])
        for i in range(j):
            H[i, j - 1] = np.dot(w, zs[i + 1])
            w -= H[i, j - 1] * us[i + 1]
        us.append(w)
        zs.append(C @ w)
        hjpj = np.sqrt(np.dot(w, zs[j + 1]))
        H[j, j - 1] = hjpj
        if np.abs(hjpj) > 1e-5:
            zs[j + 1] /= hjpj
            us[j + 1] /= hjpj
    Uq = np.stack(us[1:], axis=1)
    Hq = H[:q, :]
    return Uq, Hq, lu, piv

def compute_poles_res(Uq, Hq, C, G, Glu, Gpiv):
    eig, eigP = np.linalg.eig(Hq)
    n = Glu.shape[0]
    q = Hq.shape[0]
    e = np.zeros(n)
    e[0] = -1.
    r = lg.lu_solve((Glu, Gpiv), e)
    norm_r = np.sqrt(np.dot(r, C @ r))
    poles = 1. / eig
    residues_mat = norm_r * (Uq[:, :q] @ eigP) * eigP[0, :].reshape(1, q)
    return poles, residues_mat

def exact_solution_compatible_nodriver_nosinkcap(rc, q=4, time_step=0.01, n_steps=5000):
    C, G = build_matrix_dr(rc, 1.)
    # undo sink cap
    for i, c in enumerate(rc.sink_cell_caps):
        C[i + 2, i + 2] -= c
    # clip n only
    C = C[1:, 1:]
    G = G[1:, 1:]
    C[0, 0] = 0.
    G[0, :] = 0.
    G[0, 0] = 1.
    pdb.set_trace()
    assert C.shape == (rc.n, rc.n)
    assert G.shape == (rc.n, rc.n)
    Uq, Hq, Glu, Gpiv = ctarnoldi(C, G, q)
    poles, residues_mat = compute_poles_res(Uq, Hq, C, G, Glu, Gpiv)
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

    # C, G = build_matrix_dr(tree_rc, 1.)
    # Uq, Hq, Glu, Gpiv = ctarnoldi(C, G, 4)
    # poles, residues_mat = compute_poles_res(Uq, Hq, C, G, Glu, Gpiv)
    # eig, eigP = np.linalg.eig(Hq)

    # compute exact solution using this reduced-order model
    vs = exact_solution_compatible_nodriver_nosinkcap(tree_rc)
    from spice import spice_calc_vt
    vs_spice = spice_calc_vt(tree_rc, slew=0.,
                             time_step=0.01, n_steps=5000,
                             method='trapezoidal')
    import utils
    utils.plot(tree_rc, [('CT-Arnoldi (Order 4)', vs),
                         ('SPICE Trapezoidal', vs_spice)],
               title='CT-Arnoldi ROM')
