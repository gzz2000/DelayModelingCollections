# Gate delay computation with CT-Arnoldi reduced RC loads
# adapted from Dartu, Menezes, Pileggi, TCAD 1996

import numpy as np
import math
import pdb

from ctarnoldi import build_matrix_dr, ctarnoldi, compute_poles_res
from netlist import tree_rc
from netlist.simple_nldm import inv_x1_az_cell_rise as delay_lut, inv_x1_az_rise_trans as slew_lut

order = 4

# from liberty
# might be different for rise/fall later, be careful.
th_delay = 0.5
th_slew1, th_slew2 = 0.2, 0.8

pdb.set_trace()

# from propagation setup
input_slew = 1.01

total_cap = sum(cap for _, cap in tree_rc.grounded_caps) + sum(tree_rc.sink_cell_caps)

# adapted from eq(22) to use slew instead.
init_cap = min(total_cap, delay_lut.xs[-1])
init_slew = slew_lut.lookup(init_cap, input_slew)
init_delay = delay_lut.lookup(init_cap, input_slew)
Rd = init_slew / (init_cap * math.log(th_slew2 / th_slew1))

C, G = build_matrix_dr(tree_rc, Rd)
Uq, Hq, Glu, Gpiv = ctarnoldi(C, G, order)
poles, residues_mat = compute_poles_res(Uq, Hq, C, G, Glu, Gpiv)
residues_mat = residues_mat[1:, :]

# adapted from eq(16)-(17). initial guess of Ceff and driver
Ceff = init_cap
dt = init_slew / (th_slew2 - th_slew1)
t0 = init_delay - 0.69 * Rd * Ceff - dt / 2.

# our main task is to solve the following equation of
#  Ceff, t0, and dt.  (adapted from eq(12))
# Ipi(dt, dt) = ICeff(dt, dt, Ceff) ..... [0]
# Timeof(0.5, t0, dt) = delay(Ceff)
#  -> i.e., waveform(t0, dt, delay(Ceff)) = 0.5 ..... [1]
# Timeof(0.8, t0, dt) - Timeof(0.2, t0, dt) = slew(Ceff)
#
# introduce an auxiliary variable tr1,
# and replace the last equality with the following:
# waveform(t0, dt, tr1) = 0.2 ..... [2]
# waveform(t0, dt, tr1 + slew(Ceff)) = 0.8 ..... [3]
tr1 = init_delay - init_slew * (1 - th_slew1)

def calc_waveform(t, dt):
    if t < 0: return 0.
    t1 = max(0, t - dt)
    return (t - t1 - np.dot(residues_mat[0] * poles, np.exp(t * poles) - np.exp(t1 * poles))) / dt

def calc_waveform_grad(t, dt):
    if t < 0: return 0., 0.
    g_dt = -calc_waveform(t, dt) / dt**2
    g_t1 = 1. - np.dot(residues_mat[0] * poles**2, np.exp(t * poles))
    if t >= dt:
        g_t1 -= 1. - np.dot(residues_mat[0] * poles**2, np.exp((t - dt) * poles))
    g_t = g_t1 / dt
    return g_t, g_dt

# NR iteration
for niter in range(20):
    fval = np.zeros(4)
    jacobian = np.zeros((4, 4))

    Qpidt = np.dot(residues_mat[0], (-1. + np.exp(dt * poles) - dt * poles) / (dt * poles**2 * Rd))
    QCeffdt = Ceff + Rd * Ceff**2 * (math.exp(-dt / (Ceff * Rd))) / dt
    fval[0] = Qpidt - QCeffdt

    delay, delay_grad_x, _ = delay_lut.lookup_grad(Ceff, input_slew)
    slew, slew_grad_x, _ = slew_lut.lookup_grad(Ceff, input_slew)
    fval[1] = calc_waveform(delay - t0, dt) - th_delay
    fval[2] = calc_waveform(tr1 - t0, dt) - th_slew1
    fval[3] = calc_waveform(tr1 + slew - t0, dt) - th_slew2
    
    # if np.max(np.abs(delta)) <
    print(f'iter {niter}, Ceff={Ceff}, dt={dt}, t0={t0}, tr1={tr1}, fval={fval}')

    # df0 / dCeff
    jacobian[0, 0] = -1. - (-2. * Ceff * Rd + math.exp(-dt / (Ceff * Rd)) * (dt + 2 * Ceff * Rd)) / dt
    # df0 / ddt
    jacobian[0, 1] = np.dot(
        # Qpidt
        (1. + np.exp(dt * poles) * (-1. + dt * poles)) / (dt**2 * poles**2 * Rd) -
        # QCeffdt
        Ceff * (np.exp(-dt / (Ceff * Rd)) * (-dt - Ceff * Rd) + Ceff * Rd) / dt**2
    , residues_mat[0])

    # df{1, 2, 3}
    gwf1t, gwf1dt = calc_waveform_grad(delay - t0, dt)
    gwf2t, gwf2dt = calc_waveform_grad(tr1 - t0, dt)
    gwf3t, gwf3dt = calc_waveform_grad(tr1 + slew - t0, dt)
    
    jacobian[1, 0] = gwf1t * delay_grad_x
    jacobian[1, 1] = gwf1dt
    jacobian[1, 2] = -gwf1t
    
    jacobian[2, 1] = gwf2dt
    jacobian[2, 2] = -gwf2t
    jacobian[2, 3] = gwf2t

    jacobian[3, 0] = gwf3t * slew_grad_x
    jacobian[3, 1] = gwf3dt
    jacobian[3, 2] = -gwf3t
    jacobian[3, 3] = gwf3t
    
    delta = np.linalg.pinv(jacobian) @ fval
    Ceff -= delta[0]
    dt -= delta[1]
    t0 -= delta[2]
    tr1 -= delta[3]
    # if np.max(np.abs(delta)) <
    print(f'iter {niter} UPDATE, delta={delta}')
