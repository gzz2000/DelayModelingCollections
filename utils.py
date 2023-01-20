# plotting utility

import matplotlib
import matplotlib.pyplot as plt

# give a subplot for every rc endpoint
# plot multiple methods at once
def plot(rc, waveforms, title='Waveform Debugging'):
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(
        len(rc.endpoints), 1,
        figsize=(10, 4 * len(rc.endpoints)),
        dpi=80)
    fig.canvas.manager.set_window_title(title)
    for i in range(len(rc.endpoints)):
        eid, io = rc.endpoints[i]
        ax[i].title.set_text('{} ({})'.format(rc.names[eid], io))
        for method, vs in waveforms:
            ax[i].plot(
                [t for t, _ in vs], [v[eid] for _, v in vs],
                label=method)
        ax[i].legend()
    
    plt.tight_layout(h_pad=5)
    plt.show()
