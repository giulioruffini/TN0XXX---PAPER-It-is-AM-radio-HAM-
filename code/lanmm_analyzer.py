import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.signal import butter, filtfilt,  hilbert
from scipy.signal import welch
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import pearsonr
from lanmmv11 import get_intrinsic_params   
from scipy.signal import find_peaks
def plot_sim_results(sim_results,save_path=None,show=True):
    #document the code
    """
    Plots the results of a simulation of the LaNMM.
    """

    
    # Plot inputs
    plt.figure(figsize=(18,4))
    plt.plot(sim_results['t'], sim_results['e1_array'], label="P1 input", color='blue')
    plt.plot(sim_results['t'], sim_results['e2_array'], label="P2 input", color='red')
    plt.plot(sim_results['t'], sim_results['pv_array'], label="PV input", color='orange')
    plt.xlabel("Time (s)")
    plt.xlim(5,10)
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Inputs")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path+'inputs.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    # Plot membrane potentials.
    plt.figure(figsize=(18,6))
    plt.plot(sim_results['t'], sim_results['vP1'], label="vP1", color='blue')
    plt.plot(sim_results['t'], sim_results['vP2'], label="vP2", color='red')
    plt.plot(sim_results['t'], sim_results['vPV'], label="vPV", color='orange')
    plt.xlabel("Time (s)")
    plt.xlim(5,10)
    plt.ylabel("Membrane Potential (mV)")
    plt.title("LaNMM Simulation Results (membrane potential)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path+'v.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()



    # plot all the us
    plt.figure(figsize=(18,4))

    plt.plot(sim_results['t'], sim_results['u3'], label="u3", color='blue')
    plt.plot(sim_results['t'], sim_results['u8'], label="u8", color='red')
    plt.plot(sim_results['t'], sim_results['u14'], label="u14", color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("u (mV)")
    plt.title("u's connected to the external drive")    
    plt.xlim(5,10)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path+'u_external.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

  
        # plot all the us
    plt.figure(figsize=(18,6))
    plt.plot(sim_results['t'], sim_results['u1']+10, label="u1")
    plt.plot(sim_results['t'], sim_results['u2']+20, label="u2")
    plt.plot(sim_results['t'], sim_results['u3']+30, label="u3")
    plt.plot(sim_results['t'], sim_results['u4']+40, label="u4")
    plt.plot(sim_results['t'], sim_results['u5']+50, label="u5")
    plt.plot(sim_results['t'], sim_results['u6']+60, label="u6")
    plt.plot(sim_results['t'], sim_results['u7']+70, label="u7")
    plt.plot(sim_results['t'], sim_results['u8']+80, label="u8")
    plt.plot(sim_results['t'], sim_results['u9']+90, label="u9")
    plt.plot(sim_results['t'], sim_results['u10']+100, label="u10") 
    plt.plot(sim_results['t'], sim_results['u11']+110, label="u11")
    plt.plot(sim_results['t'], sim_results['u12']+120, label="u12")
    plt.plot(sim_results['t'], sim_results['u13']+130, label="u13")
    plt.plot(sim_results['t'], sim_results['u14']+140, label="u14")
    plt.xlabel("Time (s)")
    plt.ylabel("u (mV)")
    plt.title("u (membrane perturbation from synapses)'s")    
    plt.xlim(5,10)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path+'u.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    # Extract simulated data
    dt = sim_results['t'][1]-sim_results['t'][0]
    fs = 1 / dt  # Sampling frequency (1/dt)

    
   # Compute Welch PSD using 4-second segments
    frequencies1, psd1 = welch(sim_results['vP1'] , fs=fs, nperseg=fs*4)
    frequencies2, psd2 = welch(sim_results['vP2'] , fs=fs, nperseg=fs*4)
    frequencies, psd = welch(sim_results['vP1'] + sim_results['vP2'], fs=fs, nperseg=fs*4)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(18, 5))

    # Plot the PSD line
    ax.loglog(frequencies, psd, label='PSD (Welch)')

    # Find peaks in the PSD
    peaks, _ = find_peaks(psd, distance=10)

    # Filter peaks within the x-axis limits [0, 100] Hz
    valid_mask = (frequencies[peaks] >= 1) & (frequencies[peaks] <= 100)
    valid_peaks = peaks[valid_mask]

    # Scatter plot for valid peaks
    ax.scatter(frequencies[valid_peaks], psd[valid_peaks], color='red', label='Peaks')

    # Limit labeling to a maximum of 24 peaks:
    if len(valid_peaks) > 24:
        # Select the 12 peaks with highest PSD values, then sort them in increasing frequency order
        top_peaks = valid_peaks[np.argsort(psd[valid_peaks])[::-1][:12]]
        top_peaks = np.sort(top_peaks)
    else:
        top_peaks = valid_peaks

    # Label the selected peaks with their frequency values
    for peak in top_peaks:
        ax.text(frequencies[peak], psd[peak], f'{frequencies[peak]:.2f} Hz', color='red')

    # Configure axes, grid, legend, and set logarithmic scale
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Welch Power Spectral Density of vP1+vP2')
    ax.set_xlim(1, 100)
    ax.set_ylim(1e-8, 1e4)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path + 'psd.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    # --- Compute PSDs ---
    f1, P1 = welch(sim_results['vP1'], fs=fs, nperseg=fs*4)
    f2, P2 = welch(sim_results['vP2'], fs=fs, nperseg=fs*4)
    #fS, PS = welch(sim_results['vP1'] + sim_results['vP2'], fs=fs, nperseg=fs*4)
    
    # --- One figure / one axes ---
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Main lines
    #(line_sum,) = ax.plot(fS, PS, label='PSD Welch: P1+vP2', linewidth=2)
    (line_p1,)  = ax.plot(f1, P1, label='PSD Welch: P1', alpha=0.8)
    (line_p2,)  = ax.plot(f2, P2, label='PSD Welch: P2', alpha=0.8)
    
    # --- Peak detection helper ---
    def mark_peaks(f, P, label, color=None, max_peaks=24, fmin=0, fmax=100):
        # Find peaks in this PSD
        peaks, _ = find_peaks(P, distance=10)          # tune distance/prominence as needed
        mask = (f[peaks] >= fmin) & (f[peaks] <= fmax)
        sel = peaks[mask]
        if sel.size == 0:
            return
        # Keep at most top-N by height, then sort by frequency
        if sel.size > max_peaks:
            sel = sel[np.argsort(P[sel])[::-1][:max_peaks]]
            sel = np.sort(sel)
        sc = ax.scatter(f[sel], P[sel], s=30, marker='o', edgecolors='none',
                        label=f'Peaks: {label}', color=color)
        # Optional labels (comment out if too busy)
        for k in sel:
            ax.text(f[k], P[k], f'{f[k]:.2f} Hz', fontsize=9, va='bottom')
    
    # --- Mark peaks (toggle which series to annotate) ---
    #mark_peaks(fS, PS, 'sum (vP1+vP2)', color=line_sum.get_color(), max_peaks=24)
    # If you also want per-component peaks, uncomment:
    mark_peaks(f1, P1, 'P1', color=line_p1.get_color(), max_peaks=24)
    mark_peaks(f2, P2, 'P2', color=line_p2.get_color(), max_peaks=24)
    
    # --- Cosmetics ---
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Welch PSD: P1 and P2')
    ax.set_xlim(0, 100)
    ax.set_yscale('log')
    # If you prefer fixed limits, keep yours; otherwise auto-scale with a margin:
    # ymin = max(1e-12, np.nanmin([P1[P1>0].min(), P2[P2>0].min(), PS[PS>0].min()]) * 0.5)
    # ymax = np.nanmax([P1.max(), P2.max(), PS.max()]) * 1.5
    # ax.set_ylim(ymin, ymax)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    
    # Save/show if desired
    if save_path:
        plt.savefig(save_path + 'psd_all.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    #%%%%%%%%%

        # --- Drop DC (f=0) for log-x ---
    def positive_freqs(f, P, fmin=1, fmax=100):
        m = (f >= fmin) & (f <= fmax)
        return f[m], P[m]
    
    f1, P1 = positive_freqs(f1, P1, fmin=0.5, fmax=100)
    f2, P2 = positive_freqs(f2, P2, fmin=0.5, fmax=100)
    # fS, PS = positive_freqs(fS, PS, fmin=0.5, fmax=100)
    
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # --- Lines (log–log) ---
    #(line_sum,) = ax.loglog(fS, PS, label='Welch: vP1+vP2', linewidth=2)
    (line_p1,)  = ax.loglog(f1, P1, label='Welch: P1', alpha=0.85)
    (line_p2,)  = ax.loglog(f2, P2, label='Welch: P2', alpha=0.85)
    
 
    
    # mark_peaks(fS, PS, 'sum', color=line_sum.get_color(), max_peaks=12)
    # If desired for components too:
    mark_peaks(f1, P1, 'vP1', color=line_p1.get_color(), max_peaks=8)
    mark_peaks(f2, P2, 'vP2', color=line_p2.get_color(), max_peaks=8)
    
    # --- Cosmetics ---
    ax.set_xlabel('Frequency (Hz, log)')
    ax.set_ylabel('Power Spectral Density (log)')
    ax.set_title('Welch PSD (log–log): P1, P2')
    ax.set_xlim(1, 100)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + 'psd_all_loglog.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
def analyze_sweep_couplings(all_results, alpha_band=(8, 12), gamma_band=(30, 50), dt=0.001):
    """
    Computes coupling metrics (S2E and E2E) for each simulation in a parameter sweep.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of simulation results with keys (mu_p1, mu_p2).
    alpha_band : tuple, optional
        Frequency band for alpha.
    gamma_band : tuple, optional
        Frequency band for gamma.
    dt : float, optional
        Sampling period.
    
    Returns
    -------
    dict
        Dictionary mapping (m1, m2) to coupling metrics:
        {'r_s2e': ..., 'r_e2e': ..., 'p_s2e': ..., 'p_e2e': ...}.
    """
    couplings = {}
    for (key, result) in all_results.items():
        r_s2e, r_e2e, p_s2e, p_e2e = compute_s2e_e2e(result, alpha_band=alpha_band, gamma_band=gamma_band)
        couplings[key] = {'r_s2e': r_s2e, 'r_e2e': r_e2e, 'p_s2e': p_s2e, 'p_e2e': p_e2e}
    return couplings





##############################
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    Applies a bandpass Butterworth filter to the signal.
    
    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sampling frequency.
    lowcut : float
        Low cutoff frequency.
    highcut : float
        High cutoff frequency.
    order : int, optional
        Order of the filter.
    
    Returns
    -------
    array-like
        Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def compute_s2e_e2e(result, alpha_band=(8,12), gamma_band=(30,50)):
    """
    Computes S2E and E2E coupling metrics from simulation results.
    
    S2E: Pearson correlation between the bandpassed alpha signal of P1 and
         the envelope of the gamma band signal of P2.
    E2E: Pearson correlation between the envelopes of P1 (alpha) and P2 (gamma).
    
    Parameters
    ----------
    result : dict
        Simulation result dictionary with keys 't', 'vP1', and 'vP2'.
    alpha_band : tuple, optional
        Frequency band for alpha (default: (8, 12)).
    gamma_band : tuple, optional
        Frequency band for gamma (default: (30, 50)).
    
    Returns
    -------
    tuple
        (r_s2e, r_e2e, pval_s2e, pval_e2e)
    """
    t = result['t']
    vP1 = result['vP1']
    vP2 = result['vP2']
    
    dt_val = t[1] - t[0]
    fs = 1.0 / dt_val

    p1_alpha = bandpass_filter(vP1, fs, alpha_band[0], alpha_band[1], order=4)
    p2_gamma = bandpass_filter(vP2, fs, gamma_band[0], gamma_band[1], order=4)

    env_p1 = np.abs(hilbert(p1_alpha))
    env_p2 = np.abs(hilbert(p2_gamma))

    trim_samples = int(0.5 * fs)
    env_p1 = env_p1[trim_samples:-trim_samples]
    env_p2 = env_p2[trim_samples:-trim_samples]
    p1_alpha = p1_alpha[trim_samples:-trim_samples]

    r_s2e, p_s2e = pearsonr(p1_alpha, env_p2)
    r_e2e, p_e2e = pearsonr(env_p1, env_p2)

    return r_s2e, r_e2e, p_s2e, p_e2e

def plot_scatter_couplings(envP1_alpha, envP2_gamma, vP1_alpha):
    """
    Creates scatter plots for coupling metrics:
      - Left: Envelope of P1 (alpha) vs. envelope of P2 (gamma) (AAC).
      - Right: Bandpassed P1 (alpha) vs. envelope of P2 (gamma) (S2E).
    
    Parameters
    ----------
    envP1_alpha : array-like
        Amplitude envelope of P1's alpha band.
    envP2_gamma : array-like
        Amplitude envelope of P2's gamma band.
    vP1_alpha : array-like
        Bandpassed alpha signal of P1.
    """
    x_envelope = envP1_alpha
    y_envelope = envP2_gamma
    r_aac, p_aac = pearsonr(x_envelope, y_envelope)
    coeffs = np.polyfit(x_envelope, y_envelope, deg=1)
    fit_line = np.polyval(coeffs, x_envelope)

    x_signal = vP1_alpha
    y_env = envP2_gamma
    r_s2e, p_s2e = pearsonr(x_signal, y_env)
    coeffs2 = np.polyfit(x_signal, y_env, deg=1)
    fit_line2 = np.polyval(coeffs2, x_signal)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    axes[0].plot(x_envelope, y_envelope, 'o', alpha=0.4, label='Envelope data')
    axes[0].plot(x_envelope, fit_line, 'k--', label='Fit line')
    axes[0].set_xlabel("Alpha Envelope (P1)")
    axes[0].set_ylabel("Gamma Envelope (P2)")
    axes[0].set_title(f"Envelope-vs-Envelope: r={r_aac:.3f}, p={p_aac:.3g}")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x_signal, y_env, 'o', alpha=0.4, label='Raw alpha vs gamma env')
    axes[1].plot(x_signal, fit_line2, 'k--', label='Fit line')
    axes[1].set_xlabel("Alpha Signal (bandpassed P1)")
    axes[1].set_ylabel("Gamma Envelope (P2)")
    axes[1].set_title(f"Alpha-vs-Gamma Envelope: r={r_s2e:.3f}, p={p_s2e:.3g}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def inspect_couplings(result, t_start=1.0, t_end=30.0,
                      alpha_band=(8, 12), gamma_band=(30, 50),
                      trim_duration=0.5):
    """
    Visualizes coupling by plotting raw signals, bandpassed signals with envelopes,
    and scatter plots for AAC and S2E.
    
    Parameters
    ----------
    result : dict
        Simulation result dictionary with keys 't', 'vP1', and 'vP2'.
    t_start : float, optional
        Start time for the snippet.
    t_end : float, optional
        End time for the snippet.
    alpha_band : tuple, optional
        Frequency band for alpha.
    gamma_band : tuple, optional
        Frequency band for gamma.
    trim_duration : float, optional
        Duration to trim from the beginning and end (in seconds).
    """
    from scipy.signal import hilbert
    t = result['t']
    vP1 = result['vP1']
    vP2 = result['vP2']
    
    idx_mask = (t >= t_start) & (t <= t_end)
    t_snip = t[idx_mask]
    vP1_snip = vP1[idx_mask]
    vP2_snip = vP2[idx_mask]
    
    dt_val = t[1] - t[0]
    fs = 1.0 / dt_val
    
    plt.figure(figsize=(12, 4))
    plt.plot(t_snip, vP1_snip, 'b', label='vP1 (raw)')
    plt.plot(t_snip, vP2_snip, 'r', label='vP2 (raw)')
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("Raw Time Series")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    vP1_alpha = bandpass_filter(vP1_snip, fs, alpha_band[0], alpha_band[1], order=2)
    vP2_gamma = bandpass_filter(vP2_snip, fs, gamma_band[0], gamma_band[1], order=2)
    
    envP1_alpha = np.abs(hilbert(vP1_alpha))
    envP2_gamma = np.abs(hilbert(vP2_gamma))
    
    trim_samples = int(trim_duration * fs)
    vP1_alpha = vP1_alpha[trim_samples:-trim_samples]
    vP2_gamma = vP2_gamma[trim_samples:-trim_samples]
    envP1_alpha = envP1_alpha[trim_samples:-trim_samples]
    envP2_gamma = envP2_gamma[trim_samples:-trim_samples]
    t_snip = t_snip[trim_samples:-trim_samples]
    
    plt.figure(figsize=(12, 4))
    plt.plot(t_snip, vP1_alpha, 'b--', label=f'P1 alpha {alpha_band} Hz')
    plt.plot(t_snip, envP1_alpha, 'k', lw=2, label='P1 alpha envelope')
    plt.plot(t_snip, vP2_gamma, 'r--', label=f'P2 gamma {gamma_band} Hz')
    plt.plot(t_snip, envP2_gamma, 'g', lw=2, label='P2 gamma envelope')
    plt.xlabel("Time (s)")
    plt.ylabel("Bandpassed signals & envelopes")
    plt.title("Bandpassed & Envelopes (Trimmed)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_scatter_couplings(envP1_alpha, envP2_gamma, vP1_alpha)

def analyze_sweep_couplings(all_results, alpha_band=(8, 12), gamma_band=(30, 50), dt=0.001):
    """
    Computes coupling metrics (S2E and E2E) for each simulation in a parameter sweep.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of simulation results with keys (mu_p1, mu_p2).
    alpha_band : tuple, optional
        Frequency band for alpha.
    gamma_band : tuple, optional
        Frequency band for gamma.
    dt : float, optional
        Sampling period.
    
    Returns
    -------
    dict
        Dictionary mapping (m1, m2) to coupling metrics:
        {'r_s2e': ..., 'r_e2e': ..., 'p_s2e': ..., 'p_e2e': ...}.
    """
    couplings = {}
    for (key, result) in all_results.items():
        r_s2e, r_e2e, p_s2e, p_e2e = compute_s2e_e2e(result, alpha_band=alpha_band, gamma_band=gamma_band)
        couplings[key] = {'r_s2e': r_s2e, 'r_e2e': r_e2e, 'p_s2e': p_s2e, 'p_e2e': p_e2e}
    return couplings

def plot_coupling_heatmaps(couplings, title=None, save_path=None,
                           nominal_mu_p1=200, nominal_mu_p2=90, add_colorbar=True):
    """
    Plots 2D heatmaps for coupling metrics (S2E and E2E) with interpolation for smoother visuals.
    
    Parameters
    ----------
    couplings : dict
        Dictionary of coupling metrics with keys (mu_p1, mu_p2).
    title : str, optional
        Title for the entire plot.
    save_path : str, optional
        File path to save the plot.
    nominal_mu_p1 : float, optional
        Nominal mu_p1 value for marker. If None, no marker is plotted.
    nominal_mu_p2 : float, optional
        Nominal mu_p2 value for marker. If None, no marker is plotted.
    add_colorbar : bool, optional
        Whether to add a colorbar.
    """
    mu_p1_values = sorted(set(key[0] for key in couplings.keys()))
    mu_p2_values = sorted(set(key[1] for key in couplings.keys()))

    s2e_array = np.full((len(mu_p2_values), len(mu_p1_values)), np.nan)
    e2e_array = np.full((len(mu_p2_values), len(mu_p1_values)), np.nan)
    
    for i, m2 in enumerate(mu_p2_values):
        for j, m1 in enumerate(mu_p1_values):
            s2e_array[i, j] = couplings.get((m1, m2), {}).get('r_s2e', np.nan)
            e2e_array[i, j] = couplings.get((m1, m2), {}).get('r_e2e', np.nan)

    fine_grid_size = 500
    mu_p1_fine = np.linspace(mu_p1_values[0], mu_p1_values[-1], fine_grid_size)
    mu_p2_fine = np.linspace(mu_p2_values[0], mu_p2_values[-1], fine_grid_size)

    s2e_spline = RectBivariateSpline(mu_p2_values, mu_p1_values, s2e_array)
    e2e_spline = RectBivariateSpline(mu_p2_values, mu_p1_values, e2e_array)

    s2e_fine = s2e_spline(mu_p2_fine, mu_p1_fine)
    e2e_fine = e2e_spline(mu_p2_fine, mu_p1_fine)

    MU1_fine, MU2_fine = np.meshgrid(mu_p1_fine, mu_p2_fine)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    if title:
        fig.suptitle(title, fontsize=24, y=0.95)

    cmap = plt.cm.seismic
    norm = Normalize(vmin=-0.7, vmax=0.7)

    im1 = axes[0].imshow(s2e_fine, extent=[mu_p1_fine.min(), mu_p1_fine.max(), mu_p2_fine.min(), mu_p2_fine.max()],
                          origin='lower', aspect='equal', cmap=cmap, norm=norm, interpolation='bilinear')
    axes[0].set_title("SEC Coupling", fontsize=20)
    axes[0].set_xlabel(r"$\mu_{p1}$ (Hz)", fontsize=18)
    axes[0].set_ylabel(r"$\mu_{p2}$ (Hz)", fontsize=18)
    axes[0].tick_params(labelsize=16)
    # Plot markers only if nominal_mu_p1 and nominal_mu_p2 are provided.
    if (nominal_mu_p1 is not None) and (nominal_mu_p2 is not None):
        axes[0].plot(nominal_mu_p1, nominal_mu_p2, 'ko', markersize=15,
                     markeredgewidth=1.5, markeredgecolor='k', markerfacecolor='w')
        axes[0].plot(270, nominal_mu_p2, 'ko', markersize=15,
                     markeredgewidth=1.5, markeredgecolor='k', markerfacecolor='k')
    
    im2 = axes[1].imshow(e2e_fine, extent=[mu_p1_fine.min(), mu_p1_fine.max(), mu_p2_fine.min(), mu_p2_fine.max()],
                          origin='lower', aspect='equal', cmap=cmap, norm=norm, interpolation='bilinear')
    axes[1].set_title("EEC Coupling", fontsize=20)
    axes[1].set_xlabel(r"$\mu_{p1}$ (Hz)", fontsize=18)
    axes[1].set_ylabel(r"$\mu_{p2}$ (Hz)", fontsize=18)
    axes[1].tick_params(labelsize=16)
    if (nominal_mu_p1 is not None) and (nominal_mu_p2 is not None):
        axes[1].plot(nominal_mu_p1, nominal_mu_p2, 'ko', markersize=15,
                     markeredgewidth=1.5, markeredgecolor='k', markerfacecolor='w')
        axes[1].plot(270, nominal_mu_p2, 'ko', markersize=15,
                     markeredgewidth=1.5, markeredgecolor='k', markerfacecolor='k')
    
    if add_colorbar:
        cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
        cb.set_label("Correlation Coefficient", fontsize=18)
        cb.ax.tick_params(labelsize=16)

    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.95, wspace=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    plt.show()

def compute_band_power(signal, fs, freq_band=(8,12), method='bandpass_hilbert'):
    """
    Computes the average band power in a specified frequency band.
    
    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sampling frequency.
    freq_band : tuple, optional
        Frequency band (low, high) for power calculation.
    method : {'bandpass_hilbert', 'welch'}, optional
        Method to compute band power.
    
    Returns
    -------
    float
        Computed band power.
    """
    if method == 'bandpass_hilbert':
        sig_filt = bandpass_filter(signal, fs, freq_band[0], freq_band[1], order=4)
        env = np.abs(hilbert(sig_filt))
        power_val = np.mean(env**2)
        return power_val
    elif method == 'welch':
        fvals, psd = welch(signal, fs=fs, nperseg=1024)
        mask = (fvals >= freq_band[0]) & (fvals <= freq_band[1])
        band_power = np.trapz(psd[mask], x=fvals[mask])
        return band_power
    else:
        raise ValueError("Unknown method for compute_band_power")

def analyze_sweep_power(all_results, alpha_band=(8,12), gamma_band=(30,50),
                        dt=0.001, method='bandpass_hilbert'):
    """
    Computes band power metrics for P1 and P2 over a parameter sweep.
    
    Parameters
    ----------
    all_results : dict
        Simulation results with keys (mu_p1, mu_p2).
    alpha_band : tuple, optional
        Frequency band for alpha.
    gamma_band : tuple, optional
        Frequency band for gamma.
    dt : float, optional
        Sampling period.
    method : str, optional
        Method to compute band power ('bandpass_hilbert' or 'welch').
    
    Returns
    -------
    dict
        Dictionary with keys (m1, m2) mapping to power metrics:
        {'p1_alpha', 'p1_gamma', 'p2_alpha', 'p2_gamma'}.
    """
    fs = 1.0 / dt
    power_results = {}

    for (m1, m2), data in all_results.items():
        vP1 = data['vP1']
        vP2 = data['vP2']

        p1_alpha = compute_band_power(vP1, fs, freq_band=alpha_band, method=method)
        p1_gamma = compute_band_power(vP1, fs, freq_band=gamma_band, method=method)
        p2_alpha = compute_band_power(vP2, fs, freq_band=alpha_band, method=method)
        p2_gamma = compute_band_power(vP2, fs, freq_band=gamma_band, method=method)

        power_results[(m1, m2)] = {'p1_alpha': p1_alpha,
                                   'p1_gamma': p1_gamma,
                                   'p2_alpha': p2_alpha,
                                   'p2_gamma': p2_gamma}
    return power_results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RectBivariateSpline, griddata

def plot_power_heatmaps_bottom_cbar(power_results, title=None, log_scale=False, plot_dots =True,
                                    vmin_val=None, vmax_val=None, save_path=None, floor_val=1e-4,
                                    nominal_mu_p1=200, nominal_mu_p2=90):
    """
    Plots 2D heatmaps of band power for:
      - P1 alpha, P1 gamma, P2 alpha, P2 gamma
    with a horizontal colorbar at the bottom.
    Incorporates interpolation for higher resolution, ensures square subplots,
    adjusts colorbar position, applies a floor to power values, and optionally saves the plot as a high-quality PNG.
    
    Parameters
    ----------
    power_results : dict
        Dict with keys (mu_p1, mu_p2) containing power metrics.
    title : str, optional
        Main title of the figure.
    log_scale : bool, optional
        If True, use logarithmic color scaling.
    vmin_val : float, optional
        Minimum value for the color scale. If None, auto-determined.
    vmax_val : float, optional
        Maximum value for the color scale. If None, auto-determined.
    save_path : str, optional
        File path to save the figure as a PNG (e.g., "power_heatmaps.png").
    floor_val : float, optional
        Floor value to set for power metrics to avoid extremely small values.
    """
    # Step 1: Extract unique mu_p1 and mu_p2 values from the keys
    mu_p1_values = sorted({k[0] for k in power_results.keys()})
    mu_p2_values = sorted({k[1] for k in power_results.keys()})

    # Step 2: Prepare lists for interpolation
    points = []
    p1_alpha = []
    p1_gamma = []
    p2_alpha = []
    p2_gamma = []

    for (m1, m2), metrics in power_results.items():
        points.append((m1, m2))
        # Apply floor value before interpolation
        p1_alpha_val = metrics.get('p1_alpha', np.nan)
        p1_gamma_val = metrics.get('p1_gamma', np.nan)
        p2_alpha_val = metrics.get('p2_alpha', np.nan)
        p2_gamma_val = metrics.get('p2_gamma', np.nan)

        # Set floor: any value below floor_val is set to floor_val
        p1_alpha.append(p1_alpha_val if p1_alpha_val >= floor_val else floor_val)
        p1_gamma.append(p1_gamma_val if p1_gamma_val >= floor_val else floor_val)
        p2_alpha.append(p2_alpha_val if p2_alpha_val >= floor_val else floor_val)
        p2_gamma.append(p2_gamma_val if p2_gamma_val >= floor_val else floor_val)

    points = np.array(points)
    p1_alpha = np.array(p1_alpha)
    p1_gamma = np.array(p1_gamma)
    p2_alpha = np.array(p2_alpha)
    p2_gamma = np.array(p2_gamma)

    # Step 3: Define the number of points for the finer grid
    fine_grid_size = 500  # Adjust for desired smoothness

    # Create fine grid values for mu_p1 and mu_p2
    mu_p1_fine = np.linspace(mu_p1_values[0], mu_p1_values[-1], fine_grid_size)
    mu_p2_fine = np.linspace(mu_p2_values[0], mu_p2_values[-1], fine_grid_size)
    MU1_fine, MU2_fine = np.meshgrid(mu_p1_fine, mu_p2_fine)

    # Flatten the fine grid for interpolation
    fine_points = np.vstack((MU1_fine.ravel(), MU2_fine.ravel())).T

    # Step 4: Interpolate each power metric using griddata
    def interpolate_power_griddata(original_points, original_values, fine_points, method='cubic'):
        """
        Interpolates power metrics using griddata with specified method.
        
        Parameters:
        - original_points: array-like, shape (n_points, 2)
            Coordinates of the data points.
        - original_values: array-like, shape (n_points,)
            Values at the data points.
        - fine_points: array-like, shape (m_points, 2)
            Coordinates where interpolation is desired.
        - method: str, optional
            Interpolation method: 'linear', 'cubic', 'nearest'.
        
        Returns:
        - interpolated: 2D numpy array, shape (len(mu_p2_fine), len(mu_p1_fine))
        """
        # Perform interpolation
        interpolated = griddata(original_points, original_values, fine_points, method=method)
        
        # Handle any remaining NaNs by nearest interpolation
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            interpolated[nan_mask] = griddata(original_points, original_values, fine_points[nan_mask], method='nearest')
        
        # Reshape to the fine grid shape
        interpolated = interpolated.reshape(MU1_fine.shape)
        
        # Apply floor after interpolation
        interpolated = np.where(interpolated >= floor_val, interpolated, floor_val)
        
        return interpolated

    # Interpolate each power metric
    p1_alpha_fine = interpolate_power_griddata(points, p1_alpha, fine_points, method='cubic')
    p1_gamma_fine = interpolate_power_griddata(points, p1_gamma, fine_points, method='cubic')
    p2_alpha_fine = interpolate_power_griddata(points, p2_alpha, fine_points, method='cubic')
    p2_gamma_fine = interpolate_power_griddata(points, p2_gamma, fine_points, method='cubic')

    # Step 5: Determine global data range for color scaling if custom limits are not provided
    all_data = np.concatenate([
        p1_alpha_fine.ravel(), p1_gamma_fine.ravel(),
        p2_alpha_fine.ravel(), p2_gamma_fine.ravel()
    ])
    all_data = all_data[~np.isnan(all_data)]
    if len(all_data) == 0:
        raise ValueError("No valid power data found after interpolation.")

    # Decide color scale settings
    if log_scale:
        # For log scale, ensure a positive vmin
        if vmin_val is None:
            min_pos = np.min(all_data[all_data > 0]) if np.any(all_data > 0) else floor_val
            vmin = max(min_pos, floor_val)
        else:
            vmin = max(vmin_val, floor_val)
        vmax = vmax_val if vmax_val is not None else np.max(all_data)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = r"Band Power (log scale)"
    else:
        vmin = vmin_val if vmin_val is not None else np.min(all_data)
        vmax = vmax_val if vmax_val is not None else np.max(all_data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = r"Band Power"

    cmap = plt.cm.viridis

    # Step 6: Set up the plot with a figsize that accommodates square subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))  # 20 inches wide, 20 inches tall

    if title:
        fig.suptitle(title, fontsize=24, y=0.95)  # Adjust y for better spacing

    # Function to plot each heatmap with interpolation
    def plot_interpolated_heatmap(ax, data_fine, subtitle):
        im = ax.imshow(data_fine, 
                       extent=[mu_p1_fine.min(), mu_p1_fine.max(), mu_p2_fine.min(), mu_p2_fine.max()],
                       origin='lower',
                       aspect='equal',  # Ensure square subplot
                       cmap=cmap, 
                       norm=norm, 
                       interpolation='bilinear')  # Using bilinear interpolation for smoothness
        ax.set_title(subtitle, fontsize=18)
        ax.set_xlabel(r"$\mu_{p1}$ (Hz)", fontsize=18)
        ax.set_ylabel(r"$\mu_{p2}$ (Hz)", fontsize=18)
        if plot_dots:
            ax.plot(nominal_mu_p1, nominal_mu_p2, 'ko', markersize=15,
                 markeredgecolor='k', markerfacecolor='w')
            ax.plot(270, nominal_mu_p2, 'ko', markersize=15,
             markeredgecolor='k', markerfacecolor='k')
    
    
        return im

    # Plot each heatmap with interpolation
    im_p1_alpha = plot_interpolated_heatmap(axes[1, 0], p1_alpha_fine, "P1 Alpha Power")
    im_p1_gamma = plot_interpolated_heatmap(axes[1, 1], p1_gamma_fine, "P1 Gamma Power")
    im_p2_alpha = plot_interpolated_heatmap(axes[0, 0], p2_alpha_fine, "P2 Alpha Power")
    im_p2_gamma = plot_interpolated_heatmap(axes[0, 1], p2_gamma_fine, "P2 Gamma Power")

    # Step 7: Add a single horizontal colorbar below the subplots
    # Adjust the [left, bottom, width, height] to position it slightly lower
    cbar_ax = fig.add_axes([0.3, 0.00, 0.4, 0.02])  # [left, bottom, width, height]
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=18)
    cb.ax.tick_params(labelsize=16)

    # Step 8: Adjust layout manually to prevent overlap
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, wspace=0.3, hspace=0.3)

    # Step 9: Save the figure as a high-quality PNG if save_path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")

    plt.show()


#### banana
def plot_banana_fixed(all_results, scan_over='mu1', fixed_value=None, figsize=(20,20),
                      tol=1e-3, extra_title=""):
    """
    Plots a banana-like bifurcation diagram for P1 and P2 voltages while scanning 
    over one parameter (mu1 or mu2) and holding the other approximately fixed.

    Parameters:
    - all_results: dict with keys (mu1, mu2) and values containing 
                   {'t': ..., 'vP1': ..., 'vP2': ...}.
    - scan_over: str, either 'mu1' or 'mu2', indicating which parameter to vary.
    - fixed_value: float, approximate value of the parameter held fixed.
    - figsize: tuple, size of the figure.
    - tol: float, tolerance for matching the fixed parameter.
    - extra_title: str, additional string to append to the title.
    """
    
    if fixed_value is None:
        raise ValueError("Please provide a fixed_value for the parameter to hold constant.")
        
    plt.figure(figsize=figsize)

    # Depending on what we scan over, filter keys approximately.
    if scan_over == 'mu1':
        # Fix mu2 ≈ fixed_value, vary mu1.
        subset = {k: v for k, v in all_results.items() if abs(k[1] - fixed_value) < tol}
        xlabel = "mu_p1"
        fixed_label = f"mu_p2 ≈ {fixed_value} Hz"
        # Extract scanned values from keys: first element of each key.
        scan_values = sorted({k[0] for k in subset.keys()})
    elif scan_over == 'mu2':
        # Fix mu1 ≈ fixed_value, vary mu2.
        subset = {k: v for k, v in all_results.items() if abs(k[0] - fixed_value) < tol}
        xlabel = "mu_p2"
        fixed_label = f"mu_p1 ≈ {fixed_value} Hz"
        # Extract scanned values from keys: second element of each key.
        scan_values = sorted({k[1] for k in subset.keys()})
    else:
        raise ValueError("scan_over must be 'mu1' or 'mu2'")

    if not scan_values:
        raise ValueError(f"No simulation data found for the fixed value within tolerance {tol}.")

    # Plot banana style for each scanned value.
    for val in scan_values:
        # Select the simulation data corresponding to the current scan value.
        if scan_over == 'mu1':
            # Keys of form (val, approx fixed_value).
            # We find the key that best matches our criteria.
            key = next((k for k in subset if abs(k[0] - val) < tol), None)
        else:
            # Keys of form (approx fixed_value, val).
            key = next((k for k in subset if abs(k[1] - val) < tol), None)
            
        if key is None:
            continue
        
        data = all_results[key]
        # Use the scanned value as x-coordinate.
        x_coord = val
        
        plt.plot([x_coord]*len(data['vP1']), data['vP1'], 
                 'b.', alpha=0.05, markersize=1, label='P1' if val==scan_values[0] else "")
        plt.plot([x_coord]*len(data['vP2']), data['vP2'], 
                 'r.', alpha=0.05, markersize=1, label='P2' if val==scan_values[0] else "")

    plt.title(f"LaNMM Bifurcation-like Plot ({fixed_label}) {extra_title}")
    plt.xlabel(xlabel + " (Hz)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()
 


###############################################################################
# PEIX Function
###############################################################################
def peix(x):
    """
    Computes the PEIX scalar given x = r*(<V> - V0).
    
    The PEIX metric is defined as:
        PEIX(x) = -sign(x) * (4 * exp(-x) / (1 + exp(-x))^2 - 1)
    
    Parameters
    ----------
    x : float or array-like
        Normalized deviation from V0.
    
    Returns
    -------
    float or array-like
        Computed PEIX value.
    """
    return -np.sign(x) * (4 * np.exp(-x) / (1 + np.exp(-x))**2 - 1)

def sweep_peix(all_results, params=None):
    """
    Computes the PEIX metric for each simulation in a parameter sweep.
    
    For each simulation, average membrane potentials for P1 and P2 are computed and then normalized.
    PEIX is calculated using the peix() function.
    
    Parameters
    ----------
    all_results : dict
        Dictionary with keys (mu_p1, mu_p2) and simulation result dicts.
    params : dict, optional
        Parameter dictionary. If None, default parameters are used.
    
    Returns
    -------
    dict
        Dictionary mapping (mu_p1, mu_p2) to {'peix_P1': ..., 'peix_P2': ...}.
    """
    if params is None:
        params = get_intrinsic_params()
    r = params['r_slope']
    v0_default = params['v0_default']
    v0_p2 = params['v0_p2']
    
    peix_results = {}
    for (m1, m2), result in all_results.items():
        vP1 = result['vP1']
        vP2 = result['vP2']
        mean_vP1 = np.mean(vP1)
        mean_vP2 = np.mean(vP2)
        x1 = r * (mean_vP1 - v0_default)
        x2 = r * (mean_vP2 - v0_p2)
        peix_results[(m1, m2)] = {'peix_P1': peix(x1), 'peix_P2': peix(x2)}
    return peix_results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RectBivariateSpline

def plot_peix_heatmaps(peix_results, title=None, save_path=None,
                       nominal_mu_p1=200, nominal_mu_p2=90):
    """
    Creates 2D heatmaps of the PEIX metric for P1 and P2 across a parameter sweep.
    
    Parameters
    ----------
    peix_results : dict
        Dictionary with keys (mu_p1, mu_p2) and values {'peix_P1': ..., 'peix_P2': ...}.
    title : str, optional
        Overall title for the figure.
    save_path : str, optional
        File path to save the figure.
    nominal_mu_p1 : float, optional
        Nominal mu_p1 marker value. If None, no marker is plotted.
    nominal_mu_p2 : float, optional
        Nominal mu_p2 marker value. If None, no marker is plotted.
    """
    mu_p1_values = sorted(set(key[0] for key in peix_results.keys()))
    mu_p2_values = sorted(set(key[1] for key in peix_results.keys()))
    
    peix_P1_array = np.full((len(mu_p2_values), len(mu_p1_values)), np.nan)
    peix_P2_array = np.full((len(mu_p2_values), len(mu_p1_values)), np.nan)
    
    for i, m2 in enumerate(mu_p2_values):
        for j, m1 in enumerate(mu_p1_values):
            peix_P1_array[i, j] = peix_results.get((m1, m2), {}).get('peix_P1', np.nan)
            peix_P2_array[i, j] = peix_results.get((m1, m2), {}).get('peix_P2', np.nan)
    
    fine_grid_size = 500
    mu_p1_fine = np.linspace(mu_p1_values[0], mu_p1_values[-1], fine_grid_size)
    mu_p2_fine = np.linspace(mu_p2_values[0], mu_p2_values[-1], fine_grid_size)
    
    spline_P1 = RectBivariateSpline(mu_p2_values, mu_p1_values, peix_P1_array)
    spline_P2 = RectBivariateSpline(mu_p2_values, mu_p1_values, peix_P2_array)
    
    peix_P1_fine = spline_P1(mu_p2_fine, mu_p1_fine)
    peix_P2_fine = spline_P2(mu_p2_fine, mu_p1_fine)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    if title:
        fig.suptitle(title, fontsize=20, y=0.95)
    
    cmap = plt.cm.jet_r
    norm = Normalize(vmin=-1, vmax=1)
    
    im1 = axes[0].imshow(peix_P1_fine, extent=[mu_p1_fine.min(), mu_p1_fine.max(), 
                                                 mu_p2_fine.min(), mu_p2_fine.max()],
                         origin='lower', aspect='equal', cmap=cmap, norm=norm, interpolation='bilinear')
    axes[0].set_title("PEIX for P1", fontsize=20)
    axes[0].set_xlabel(r"$\mu_{p1}$ (Hz)", fontsize=18)
    axes[0].set_ylabel(r"$\mu_{p2}$ (Hz)", fontsize=18)
    if (nominal_mu_p1 is not None) and (nominal_mu_p2 is not None):
        axes[0].plot(nominal_mu_p1, nominal_mu_p2, 'ko', markersize=15,
                     markeredgecolor='k', markerfacecolor='w')
        axes[0].plot(270, nominal_mu_p2, 'ko', markersize=15,
                     markeredgecolor='k', markerfacecolor='k')
    
    im2 = axes[1].imshow(peix_P2_fine, extent=[mu_p1_fine.min(), mu_p1_fine.max(), 
                                                 mu_p2_fine.min(), mu_p2_fine.max()],
                         origin='lower', aspect='equal', cmap=cmap, norm=norm, interpolation='bilinear')
    axes[1].set_title("PEIX for P2", fontsize=20)
    axes[1].set_xlabel(r"$\mu_{p1}$ (Hz)", fontsize=18)
    axes[1].set_ylabel(r"$\mu_{p2}$ (Hz)", fontsize=18)
    if (nominal_mu_p1 is not None) and (nominal_mu_p2 is not None):
        axes[1].plot(nominal_mu_p1, nominal_mu_p2, 'ko', markersize=15,
                     markeredgecolor='k', markerfacecolor='w')
        axes[1].plot(270, nominal_mu_p2, 'ko', markersize=15,
                     markeredgecolor='k', markerfacecolor='k')
    
    cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label("PEIX", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, wspace=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    plt.show()

    # let's plot a 2d plot of the peak frequency of the power in the alpha and gamma bands

def compute_peak_frequency(signal, fs, band, nperseg=1024):
    """
    Compute the peak frequency within a specific frequency band.
    
    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sampling frequency.
    band : tuple
        (low, high) frequency band in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method.
    
    Returns
    -------
    float
        Peak frequency in Hz.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    peak_idx = np.argmax(psd[mask])
    return freqs[mask][peak_idx]

def analyze_peak_frequencies(sweep_results, fs=1000):
    """
    Analyze peak frequencies in alpha and gamma bands for a parameter sweep.
    
    Parameters
    ----------
    sweep_results : dict
        Results from run_parameter_sweep.
    fs : float, optional
        Sampling frequency (default: 1000 Hz).
    
    Returns
    -------
    dict
        Dictionary containing peak frequency data for both bands.
    """
    alpha_band = (8, 12)
    gamma_band = (30, 80)
    
    # Get unique m1 and m2 values
    m1_values = sorted(set(m1 for m1, _ in sweep_results.keys()))
    m2_values = sorted(set(m2 for _, m2 in sweep_results.keys()))
    
    # Initialize arrays for peak frequencies
    alpha_peaks = np.zeros((len(m1_values), len(m2_values)))
    gamma_peaks = np.zeros((len(m1_values), len(m2_values)))
    
    for i, m1 in enumerate(m1_values):
        for j, m2 in enumerate(m2_values):
            result = sweep_results[(m1, m2)]
            vP2 = result['vP2']  # Using vP2 for analysis of gamma band
            vP1 = result['vP1']  # Using vP1 for analysis of alpha band
            
            alpha_peaks[i, j] = compute_peak_frequency(vP1, fs, alpha_band)
            gamma_peaks[i, j] = compute_peak_frequency(vP2, fs, gamma_band)
    
    return {
        'alpha_peaks': alpha_peaks,
        'gamma_peaks': gamma_peaks,
        'm1_values': m1_values,
        'm2_values': m2_values
    }

def plot_frequency_heatmaps(freq_data, save_path=None):
    """
    Create heatmaps for peak frequencies in alpha and gamma bands.
    
    Parameters
    ----------
    freq_data : dict
        Output from analyze_peak_frequencies.
    save_path : str, optional
        Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot alpha peaks with a cool colormap
    cmap = plt.cm.cool
    im1 = ax1.imshow(freq_data['alpha_peaks'].T, origin='lower', aspect='auto',
                     extent=[min(freq_data['m1_values']), max(freq_data['m1_values']),
                             min(freq_data['m2_values']), max(freq_data['m2_values'])],
                     cmap=cmap, interpolation='bilinear')
    ax1.set_xlabel('P1 drive (Hz)')
    ax1.set_ylabel('P2 drive (Hz)')
    ax1.set_title('Peak Alpha Frequency in P1 (Hz)')
    im1.set_clim(6, 12)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    # Plot gamma peaks with a hot colormap
    cmap = plt.cm.hot
    im2 = ax2.imshow(freq_data['gamma_peaks'].T, origin='lower', aspect='auto',
                     extent=[min(freq_data['m1_values']), max(freq_data['m1_values']),
                             min(freq_data['m2_values']), max(freq_data['m2_values'])],
                     cmap=cmap, interpolation='bilinear')
    ax2.set_xlabel('P1 drive (Hz)')
    ax2.set_ylabel('P2 drive (Hz)')
    ax2.set_title('Peak Gamma Frequency in P2 (Hz)')
    im2.set_clim(30, 50)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



def max_corr_directional(x, y, max_lag, fs=None, maximize="abs", nan_policy="omit",printit=True):
    """
    Compute Pearson correlation r between x and y for all integer lags in [0, max_lag],
    and return the lag that maximizes it.

    Parameters
    ----------
    x, y : array_like
        1-D signals of equal length.
    max_lag : int
        Maximum lag (in samples) to search in positive delay direction.
    fs : float or None
        Sampling rate (Hz). If given, a 'best_lag_sec' field is included in the result.
    maximize : {'abs','pos','neg'}
        'abs'  -> maximize |r|  (default)
        'pos'  -> maximize r
        'neg'  -> minimize r  (most negative)
    nan_policy : {'omit','propagate'}
        If 'omit', pairs containing NaNs are dropped per-lag; if 'propagate', r becomes NaN.

    Returns
    -------
    out : dict
        {
          'best_lag'     : int (samples),
          'best_r'       : float,
          'best_lag_sec' : float (only if fs is provided),
          'lags'         : np.ndarray shape (2*max_lag+1,),
          'r'            : np.ndarray correlation at each lag
        }
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0.")

    N = len(x)
    lags = np.arange(0, max_lag + 1)
    r_vals = np.full(lags.shape, np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag >= 0:
            xs = x[lag:]
            ys = y[:N - lag]
        else:
            xs = x[:N + lag]
            ys = y[-lag:]

        if nan_policy == "omit":
            m = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[m]
            ys = ys[m]
            if xs.size < 2:
                r_vals[i] = np.nan
                continue
        elif nan_policy != "propagate":
            raise ValueError("nan_policy must be 'omit' or 'propagate'.")

        # Pearson r on the overlapping segment (centers & scales locally)
        # Using unbiased sample std (N-1) normalization.
        xm = xs - xs.mean()
        ym = ys - ys.mean()
        denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
        r_vals[i] = np.sum(xm * ym) / denom if denom > 0 else np.nan

    # Choose the "best" according to maximize
    if maximize == "abs":
        idx = np.nanargmax(np.abs(r_vals))
    elif maximize == "pos":
        idx = np.nanargmax(r_vals)
    elif maximize == "neg":
        idx = np.nanargmin(r_vals)
    else:
        raise ValueError("maximize must be one of {'abs','pos','neg'}.")

    out = {
        "best_lag": int(lags[idx]),
        "best_r": float(r_vals[idx]),
        "lags": lags,
        "r": r_vals
    }
    if fs is not None:
        out["best_lag_sec"] = lags[idx] / float(fs)
    
    
    if printit:
        plt.figure(figsize=(4, 2))
        plt.plot(lags,r_vals)
        plt.title("corr vs delay (lags)")
        print("best lag and r:",int(lags[idx]), float(r_vals[idx]))
    return out

    