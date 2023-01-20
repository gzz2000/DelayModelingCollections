# exact solution for the zero-slew case
# use a eigenvalue decomposition to transform the problem to
# n-1 scalar counterparts, and then solve them.

import numpy as np
from spice import build_matrix
import pdb

# want: A1 @ v + B1 @ grad{v} = 0
# let v = P @ xi
# want: A1 @ P @ xi + B1 @ P @ grad{xi} = 0
#       grad{xi} = - inv(P) @ inv(B1) @ A1 @ P @ xi
# just take P s.t. inv(P) @ inv(B1) @ A1 @ P is diagonal,
# which means P is the eigenvector matrix of inv(B1) @ A1.
# let xi_i = exp(-lambda_i * t)

def solve_zero_slew(rc, time_step=0.01, n_steps=5000, include_internal=False):
    A, B = build_matrix(rc)
    A1 = A[1:, 1:]
    B1 = B[1:, 1:]
    
    # do eigenvalue decomposition
    eig, eigP = np.linalg.eig(np.linalg.pinv(B1) @ A1)
    
    # calculate the coefficient of xi's
    # want: 1 - eigP @ diag(coeff_xi) @ xi = v
    #       v(0) = 0, v(infinity) = 1
    #    => eigP @ diag(coeff_xi) = np.ones(n - 1)
    coeff_xi = np.sum(np.linalg.pinv(eigP), axis=1)
    
    vs = [(0, np.zeros(rc.n))]
    t = 0.
    for i in range(n_steps):
        v = np.ones(rc.n)
        # the exact solution is here
        v[1:] -= eigP @ (coeff_xi * np.exp(-eig * t))
        vs.append((t, v))
        t += time_step
    if include_internal:
        return vs, (eig, eigP, coeff_xi)
    else:
        return vs

def debug_solve(rc, compare_spice=False):
    time_step = 0.01
    n_steps = 5000
    vs, info = solve_zero_slew(
        rc, time_step=time_step, n_steps=n_steps,
        include_internal=True)

    # print the exact solution
    eig, eigP, coeff_xi = info
    print_threshold = 0.01  # allow 1% error to simplify
    for eid, io in rc.endpoints:
        print(f'Endpoint {rc.names[eid]} ({io}) index {eid}:')
        if eid == 0:
            print('= 1 (Fixed voltage)')
            continue
        print('= 1', end='')
        for i in range(rc.n - 2, 0, -1):
            coeff = eigP[eid - 1, i] * coeff_xi[i]
            if abs(coeff) < print_threshold: continue
            print(f' {-coeff:+.05} Exp[{-eig[i]:+.05} t]', end='')
        print()
    
    methods = [('Exact', vs)]
    if compare_spice:
        from spice import spice_calc_vt
        vs_spice = spice_calc_vt(
            rc, slew=0.,
            time_step=time_step, n_steps=n_steps,
            method='trapezoidal')
        methods.append(('SPICE Trapezoidal', vs_spice))

    import utils
    utils.plot(rc, methods, title='Exact Solutions')

if __name__ == '__main__':
    from netlist import tree_rc
    debug_solve(tree_rc, compare_spice=True)
