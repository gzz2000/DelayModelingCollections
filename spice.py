# a pure SPICE simulation
# modified nodal analysis (MNA) - actually, not modified at all, pure NA.
# input is supposed to be a ramp shape _/- with a specified 0-100% slew.

import numpy as np
from netlist import tree_rc
import pdb

# build matrix A, B, s. t. the voltage vector v satisfies
# Av + Bv' = u,  where u is [<root voltage input>, 0^(n-1)].transpose().
# here, the L.H.S. Av + Bv' means "net (total) out current" except the 0th line.
def build_matrix(rc):
    assert rc.endpoints[0] == (0, 'O'), 'We assume root to be #0 node'
    A = np.zeros((rc.n, rc.n))
    B = np.zeros((rc.n, rc.n))
    A[0, 0] = 1
    for i, c in rc.grounded_caps:
        if i != 0:
            B[i, i] += c
    for i1, i2, c in rc.coupling_caps:
        if i1 != 0:
            B[i1, i1] += c
            B[i1, i2] -= c
        if i2 != 0:
            B[i2, i2] += c
            B[i2, i1] -= c
    for i1, i2, r in rc.ress:
        assert r >= 0.00001, 'Too small resistance'
        invr = 1. / r
        if i1 != 0:
            A[i1, i1] += invr
            A[i1, i2] -= invr
        if i2 != 0:
            A[i2, i2] += invr
            A[i2, i1] -= invr
    return A, B

def calc_vt(rc, slew=0., time_step=0.01, n_steps=5000, method='backward_euler'):
    assert method in ['forward_euler', 'backward_euler', 'trapezoidal']
    A, B = build_matrix(rc)
    t = 0.
    v = np.zeros(rc.n)
    v[0] = 0.
    u = np.zeros(rc.n)
    # inversion is not the most efficient way to do this.
    # better use gaussian elimination for these sparse matrices.
    if method == 'forward_euler':
        invB = np.linalg.pinv(B)
    elif method == 'backward_euler':
        invApBdt = np.linalg.pinv(A + B / time_step)
    else:
        invAd2pBdt = np.linalg.pinv(A / 2. + B / time_step)
        BdtmAd2 = B / time_step - A / 2.
    vs = [(0, v)]
    for i in range(n_steps):
        if t >= slew: u[0] = 1.
        else: u[0] = t / slew
        if method == 'forward_euler':
            v = v + (invB * time_step) @ (u - A @ v)
        elif method == 'backward_euler':
            v = invApBdt @ (u + B @ v / time_step)
        else:
            v = invAd2pBdt @ (u + BdtmAd2 @ v)
        v[0] = u[0]   # ugly fix?
        vs.append((t, v))
        t += time_step
        if np.max(np.abs(v)) > 2.:
            print(f'method {method} diverges at iteration {i} time {t}')
            break
    return vs

def debug_calc(rc):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    n_steps = 5000
    slew = 6
    vs_fe = calc_vt(rc, slew=slew, n_steps=n_steps, method='forward_euler')
    vs_be = calc_vt(rc, slew=slew, n_steps=n_steps, method='backward_euler')
    vs_tp = calc_vt(rc, slew=slew, n_steps=n_steps, method='trapezoidal')
    # pdb.set_trace()

    fig, ax = plt.subplots(
        len(rc.endpoints), 1,
        figsize=(10, 4 * len(rc.endpoints)),
        dpi=80)
    fig.canvas.manager.set_window_title('SPICE Waveforms')

    for i in range(len(rc.endpoints)):
        eid, io = rc.endpoints[i]
        ax[i].title.set_text('{} ({})'.format(rc.names[eid], io))
        for method, vs in [
                ('Forward Euler', vs_fe),
                ('Backward Euler', vs_be),
                ('Trapezoidal', vs_tp)]:
            if len(vs) < n_steps:
                method += ' (Diverges)'
            ax[i].plot(
                [t for t, _ in vs], [v[eid] for _, v in vs],
                label=method)
        ax[i].legend()

    plt.tight_layout(h_pad=5)
    plt.show()

if __name__ == '__main__':
    debug_calc(tree_rc)
