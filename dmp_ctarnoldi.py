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

# from propagation setup
input_slew = 1.01

total_cap = sum(cap for _, cap in tree_rc.grounded_caps) + sum(tree_rc.sink_cell_caps)
print('total_cap', total_cap)

# adapted from eq(22) to use slew instead.
init_cap = min(total_cap, delay_lut.xs[-1])
init_slew = slew_lut.lookup(init_cap, input_slew)
init_delay = delay_lut.lookup(init_cap, input_slew)
Rd = init_slew / (init_cap * math.log(th_slew2 / th_slew1)) / 4.

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

# assumes t0 = 0 here
def calc_waveform(t, dt, node_id=0):
    if t < 0: return 0.
    t1 = max(0, t - dt)
    return (t - t1 - np.dot(residues_mat[node_id] / poles, np.exp(t * poles) - np.exp(t1 * poles))) / dt

def calc_waveform_grad(t, dt, node_id=0):
    if t < 0: return 0., 0.
    # some repeated calculation can be avoided here.
    g_dt = -calc_waveform(t, dt, node_id=node_id) / dt
    g_t = (1. - np.dot(residues_mat[node_id], np.exp(t * poles))) / dt
    if t >= dt:
        g_left = (1. - np.dot(residues_mat[node_id], np.exp((t - dt) * poles))) / dt
        g_t -= g_left
        g_dt += g_left
    return g_t, g_dt

def calc_fval(Ceff, dt, t0, tr1):
    fval = np.zeros(4, dtype=np.float32)
    
    Qpidt = np.dot(residues_mat[0], (-1. + np.exp(dt * poles) - dt * poles) / (dt * poles**2 * Rd))
    QCeffdt = Ceff + Rd * Ceff**2 * (-1. + math.exp(-dt / (Ceff * Rd))) / dt
    fval[0] = Qpidt - QCeffdt

    delay = delay_lut.lookup(Ceff, input_slew)
    slew = slew_lut.lookup(Ceff, input_slew)
    
    fval[1] = calc_waveform(delay - t0, dt) - th_delay
    fval[2] = calc_waveform(tr1 - t0, dt) - th_slew1
    fval[3] = calc_waveform(tr1 + slew - t0, dt) - th_slew2

    return fval, delay, slew

def calc_jacobian(Ceff, dt, t0, tr1):
    delay, delay_grad_x, _ = delay_lut.lookup_grad(Ceff, input_slew)
    slew, slew_grad_x, _ = slew_lut.lookup_grad(Ceff, input_slew)

    jacobian = np.zeros((4, 4), dtype=np.float32)
    
    # df0 / dCeff = - dQCeffdt / dCeff
    jacobian[0, 0] = -(dt - 2. * Ceff * Rd + math.exp(-dt / (Ceff * Rd)) * (dt + 2 * Ceff * Rd) / dt)
    # df0 / ddt
    jacobian[0, 1] = (
        # dQpidt / ddt
        np.dot((1. + np.exp(dt * poles) * (-1. + dt * poles)) / (dt**2 * poles**2 * Rd), residues_mat[0]) -
        # dQCeffdt / ddt
        Ceff * (np.exp(-dt / (Ceff * Rd)) * (-dt - Ceff * Rd) + Ceff * Rd) / dt**2
    )

    # df{1, 2, 3}
    gwf1t, gwf1dt = calc_waveform_grad(delay - t0, dt)
    gwf2t, gwf2dt = calc_waveform_grad(tr1 - t0, dt)
    gwf3t, gwf3dt = calc_waveform_grad(tr1 + slew - t0, dt)

    # df / ddt
    jacobian[1, 1] = gwf1dt
    jacobian[2, 1] = gwf2dt
    jacobian[3, 1] = gwf3dt

    # df / dt0
    jacobian[1, 2] = -gwf1t
    jacobian[2, 2] = -gwf2t
    jacobian[3, 2] = -gwf3t

    # df / dCeff
    jacobian[1, 0] = gwf1t * delay_grad_x
    jacobian[3, 0] = gwf3t * slew_grad_x

    # df / dtr1
    jacobian[2, 3] = gwf2t
    jacobian[3, 3] = gwf3t

    return jacobian

# given initial values
def debug_fval_slider(Ceff, dt, t0, tr1):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    matplotlib.use('TkAgg')

    fig, (ax_fval, ax) = plt.subplots(2)
    t = np.linspace(-50, 300, 5000, dtype=np.float32)
    
    def calc_waveform_vs(t, dt, t0):
        t = np.maximum(0., t - t0)
        return np.minimum(1., t / dt)
    
    def calc_waveform_driver(t, dt, t0):
        t = np.maximum(0., t - t0)
        t1 = np.maximum(0., t - dt)
        return (t - t1 - (residues_mat[0] / poles) @ (
            np.exp(poles.reshape(-1, 1) * t.reshape(1, -1)) -
            np.exp(poles.reshape(-1, 1) * t1.reshape(1, -1)))) / dt

    def format_fval_text(Ceff, dt, t0, tr1, waveform_driver):
        fval, delay, slew = calc_fval(Ceff, dt, t0, tr1)
        slew_l = np.searchsorted(waveform_driver, th_slew1)
        slew_r = np.searchsorted(waveform_driver, th_slew2)
        slew_waveform = t[slew_r] - t[slew_l]
        return (
            f'[0] 0 == Qpidt - QCeffdt = {fval[0]}\n'
            f'[1] 0 == waveform(delay-t0) - {th_delay} = {fval[1]}\n'
            f'[2] 0 == waveform(tr1-t0) - {th_slew1} = {fval[2]}\n'
            f'[3] 0 == waveform(tr1+slew-t0) - {th_slew2} = {fval[3]}\n'
            f'    delay = {delay}, slew = {slew}\n'
            f'    current slew = {slew_waveform}'
        )
    
    line_vs, = ax.plot(t, calc_waveform_vs(t, dt, t0), label='vs')
    waveform_driver = calc_waveform_driver(t, dt, t0)
    line_driver, = ax.plot(t, waveform_driver, label='driver')
    ax.legend()
    fig.subplots_adjust(top=1.2, bottom=0.3)

    ax_fval.axis('off')
    text_fval = ax_fval.text(0., 0.05, format_fval_text(Ceff, dt, t0, tr1, waveform_driver), va='center')

    Ceff_slider = Slider(
        ax=fig.add_axes([0.1, 0.2, 0.75, 0.03]),
        label='Ceff', valmin=0., valmax=Ceff*2., valinit=Ceff)
    dt_slider = Slider(
        ax=fig.add_axes([0.1, 0.15, 0.75, 0.03]),
        label='dt', valmin=0., valmax=300., valinit=dt)
    t0_slider = Slider(
        ax=fig.add_axes([0.1, 0.1, 0.75, 0.03]),
        label='t0', valmin=-100., valmax=300., valinit=t0)
    tr1_slider = Slider(
        ax=fig.add_axes([0.1, 0.05, 0.75, 0.03]),
        label='tr1', valmin=-100., valmax=300., valinit=tr1)
    
    def update_info(val):
        Ceff, dt, t0, tr1 = Ceff_slider.val, dt_slider.val, t0_slider.val, tr1_slider.val
        line_vs.set_ydata(calc_waveform_vs(t, dt, t0))
        waveform_driver = calc_waveform_driver(t, dt, t0)
        line_driver.set_ydata(waveform_driver)
        text_fval.set_text(format_fval_text(Ceff, dt, t0, tr1, waveform_driver))
        fig.canvas.draw_idle()
        
    for slider in [Ceff_slider, dt_slider, t0_slider, tr1_slider]:
        slider.on_changed(update_info)

    plt.show()

def test_NR(Ceff, dt, t0, tr1):
    # NR iteration
    for niter in range(500):
        print('========================================================')
        fval, delay, slew = calc_fval(Ceff, dt, t0, tr1)

        # if np.max(np.abs(delta)) <
        print(f'iter {niter}, Ceff={Ceff}, dt={dt}, t0={t0}, tr1={tr1}, fval={fval}')
        print(f'iter {niter} test delay={delay}, slew={slew}')
        # pdb.set_trace()

        jacobian = calc_jacobian(Ceff, dt, t0, tr1)
        print('jacobian', jacobian)

        delta = np.linalg.pinv(jacobian) @ fval
        Ceff -= delta[0]
        dt -= delta[1]
        t0 -= delta[2]
        tr1 -= delta[3]
        # if np.max(np.abs(delta)) <
        print(f'iter {niter} UPDATE, delta={delta}')

        dt = max(dt, 0.01)

def calc_f123val(delay, slew, dt, t0, tr1):
    fval = np.zeros(3, dtype=np.float32)
    fval[0] = calc_waveform(delay - t0, dt) - th_delay
    fval[1] = calc_waveform(tr1 - t0, dt) - th_slew1
    fval[2] = calc_waveform(tr1 + slew - t0, dt) - th_slew2
    return fval

def calc_f123jacobian(delay, slew, dt, t0, tr1):
    jacobian = np.zeros((3, 3), dtype=np.float32)
    
    # df{1, 2, 3}
    gwf1t, gwf1dt = calc_waveform_grad(delay - t0, dt)
    gwf2t, gwf2dt = calc_waveform_grad(tr1 - t0, dt)
    gwf3t, gwf3dt = calc_waveform_grad(tr1 + slew - t0, dt)

    # df / ddt
    jacobian[0, 0] = gwf1dt
    jacobian[1, 0] = gwf2dt
    jacobian[2, 0] = gwf3dt

    # df / dt0
    jacobian[0, 1] = -gwf1t
    jacobian[1, 1] = -gwf2t
    jacobian[2, 1] = -gwf3t

    # df / dtr1
    jacobian[1, 2] = gwf2t
    jacobian[2, 2] = gwf3t

    return jacobian

def test_Ceff_smallNR(Ceff, dt, t0, tr1):
    delay_old = None  # detect convergence
    slew_old = None   # detect convergence
    for ceff_i in range(20):
        delay = delay_lut.lookup(Ceff, input_slew)
        slew = slew_lut.lookup(Ceff, input_slew)
        print('===============================')
        print(f'Ceff iter {ceff_i}: Ceff={Ceff}, delay={delay}, slew={slew}')
        if delay_old is not None and \
           math.fabs((delay_old - delay) / delay_old) < 1e-3 and \
           math.fabs((slew_old - slew) / slew_old) < 1e-3:
            print(f'| CONVERGED (1‰). STOP.')
            break
        
        for nr_i in range(20):
            fval = calc_f123val(delay, slew, dt, t0, tr1)
            print(f'\\__ smallNR iter {nr_i}, dt={dt}, t0={t0}, tr1={tr1}, fval={fval}')
            if np.max(np.abs(fval)) < 1e-3:
                print(f'\\__ smallNR EARLY STOP.')
                break
            jacobian = calc_f123jacobian(delay, slew, dt, t0, tr1)
            delta = np.linalg.pinv(jacobian) @ fval
            dt -= delta[0]
            t0 -= delta[1]
            tr1 -= delta[2]
            print(f'\\__     UPDATE delta={delta}')
            assert dt > 0., 'Too small dt, you may have to decrease Rd.'
        else:
            print('WARN: smallNR not converged after many iters.')
            
        Qpidt = np.dot(residues_mat[0], (-1. + np.exp(dt * poles) - dt * poles) / (dt * poles**2 * Rd))
        QCeff_quad_coeff = Rd * (-1. + math.exp(-dt / (Ceff * Rd))) / dt
        Ceff_new = (-1. + math.sqrt(1 + 4. * QCeff_quad_coeff * Qpidt)) / (2. * QCeff_quad_coeff)
        print(f'| NEW Ceff={Ceff_new} (old: {Ceff})')
        assert Ceff_new < Ceff, 'Should most likely be decreasing.'
        Ceff = Ceff_new
        delay_old = delay
        slew_old = slew
    else:
        print('WARN: Ceff and delay not converged after many iters.')

    return Ceff, delay, slew, dt, t0, tr1

def calc_single_timepoint_nr(dt, node_id, v, init_t):
    t = init_t
    for i in range(20):
        f = calc_waveform(t, dt, node_id=node_id) - v
        if math.fabs(f) < 1e-3:
            print(f'timepoint of node {node_id} v {v}: converged in {i} iters')
            break
        g, _ = calc_waveform_grad(t, dt, node_id=node_id)
        t -= f / g
    else:
        print('WARN: single timepoint not converged after many iters.')
    return t
    
if __name__ == '__main__':
    # debug_fval_slider(Ceff, dt, t0, tr1)
    # test_NR(Ceff, dt, t0, tr1)
    Ceff, delay, slew, dt, t0, tr1 = test_Ceff_smallNR(Ceff, dt, t0, tr1)
    for endpoint, io in tree_rc.endpoints:
        if io != 'I': continue
        t_delay = calc_single_timepoint_nr(dt, endpoint, th_delay, delay - t0)
        t_slew1 = calc_single_timepoint_nr(dt, endpoint, th_slew1, delay - t0)
        t_slew2 = calc_single_timepoint_nr(dt, endpoint, th_slew2, delay - t0)

        print(f'Node {tree_rc.names[endpoint]} ({endpoint}): delay={t_delay + t0 - delay}, slew={t_slew2 - t_slew1}')
