
import numpy as np
import matplotlib.pyplot as plt
from lanmmv11 import run_unified_simulation
from lanmm_analyzer import plot_sim_results
from lanmmv11 import run_parameter_sweep
from lanmm_analyzer import (analyze_sweep_couplings, analyze_sweep_power, 
                            plot_coupling_heatmaps, plot_power_heatmaps_bottom_cbar, 
                            sweep_peix, plot_peix_heatmaps, 
                            analyze_peak_frequencies, plot_frequency_heatmaps)    
from lanmmv11 import generate_multiscale_noise, generate_nested_am_signal 
import os
import json
from datetime import datetime
import pickle   



def impulse_response(t, A, a):
    """
    Synaptic impulse response:
      h(t) = A * a * t * exp(-a t),  t >= 0
    """
    # For t < 0, the response is zero (we assume t >= 0 in the caller).
    return A * a * t * np.exp(-a * t)

def frequency_response(omega, A, a):
    """
    Frequency response in the jω-domain:
      H(jω) = A a / (a + j ω)^2
    Returns the magnitude |H(jω)|.
    """
    H = A*a / (a + 1j*omega)**2
    return np.abs(H)


def plot_synaptic_filter(A=3.25, a=100.0, 
                         tmax=0.1, nt=1000, 
                         fmax=50.0, nf=1000, title=None):
    """
    Plots the synaptic filter's impulse response h(t) and
    the magnitude of its frequency response |H(jω)| in Hz.

    Parameters
    ----------
    A : float
        Synaptic gain parameter.
    a : float
        Time constant parameter in 1/s (so 1/a is the main timescale).
    tmax : float
        Max time (s) for plotting the impulse response.
    nt : int
        Number of time samples in [0, tmax].
    fmax : float
        Max frequency (Hz) for plotting the frequency response.
    nf : int
        Number of frequency samples in [0, fmax].
    """
    # 1) Time domain
    t = np.linspace(0, tmax, nt)
    h_t = impulse_response(t, A, a)

    # 2) Frequency domain in Hz
    freq = np.linspace(0, fmax, nf)  # in Hz
    omega = 2.0 * np.pi * freq       # convert Hz to rad/s
    H_omega = frequency_response(omega, A, a)

    # DC gain for reference (|H(0)| = A/a).
    dc_gain = A / a

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # --- Plot impulse response ---
    axs[0].plot(t, h_t, 'b', linewidth=2)
    axs[0].set_title(f"Impulse Response: A={A}, a={a}, {title}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("h(t)")
    axs[0].grid(True)

    # --- Plot frequency response (magnitude) in Hz ---
    axs[1].plot(freq, H_omega, 'r', linewidth=2, label='|H(jω)|')
    axs[1].axhline(dc_gain, color='k', linestyle='--', 
                   label=f"DC Gain = A/a = {dc_gain:.3f}")
    axs[1].set_title(f"Magnitude of Frequency Response, {title}")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("|H(jω)|")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def run_sweep_job(intrinsic_params, driving_params, job_params):
        """
        Run a parameter sweep job with comprehensive analysis and visualization.
        
        Parameters
        ----------
        intrinsic_params : dict
            Intrinsic model parameters.
        driving_params : dict
            Driving parameters for external inputs.
        job_params : dict
            Job-specific parameters including:
                - mu_p1_values: range of values for P1 drive
                - mu_p2_values: range of values for P2 drive
                - tmax: simulation duration
                - dt: time step
                - discard: transient duration to discard
                - alpha_band: tuple for alpha frequency band
                - gamma_band: tuple for gamma frequency band
        job_title : str
            Title for the job, used in file naming.
        
        Returns
        -------
        dict
            Dictionary containing all results and analysis data.
        """
        # Create output directory with timestamp
        job_title = job_params['job_title']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{job_title}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        # Convert range objects to lists for JSON serialization
        params_dict = {
            'intrinsic_params': intrinsic_params,
            'driving_params': driving_params,
            'job_params': {
                **job_params,
                'mu_p1_values': list(job_params['mu_p1_values']),
                'mu_p2_values': list(job_params['mu_p2_values'])
            }
        }
        
        # Save parameters to JSON file
        with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
            json.dump(params_dict, f, indent=4)
        
        # Generate and plot sample noise
        t_array = np.arange(0, job_params['tmax'], job_params['dt'])
        fs = 1.0 / job_params['dt']
        

        sim_results = run_unified_simulation(intrinsic_params, driving_params,
                                     tmax=job_params['tmax'], 
                                     dt=job_params['dt'], 
                                     discard=job_params['discard'])
        plot_sim_results(sim_results,save_path=output_dir+'/')  
          
        
        # Generate sample noise for each drive
        for drive in ['e1', 'e2', 'pv']:
            drive_cfg = driving_params.get(drive, {})
            mode = drive_cfg.get('mode', 'constant')
            mu = drive_cfg.get('mu', 0.0)
            
            if mode == 'multiscale':
                noise = generate_multiscale_noise(
                    t_array, base_mean=mu, seed=driving_params.get('seed', 42),
                    fs=fs, **drive_cfg.get('multiscale_params', {}))
            elif mode == 'am':
                am_params = drive_cfg.get('am_params', {})
                noise = generate_nested_am_signal(
                    t_array,
                    carrier_freq = am_params.get('carrier_freq', 0.0),
                    envelope_band = am_params.get('envelope_band', (8,12)),
                    slow_band = am_params.get('slow_band', (0.5,2)),
                    carrier_amplitude = am_params.get('carrier_amplitude', 1.0),
                    mod_index_slow = am_params.get('mod_index_slow', 0.5),
                    mod_index_fast = am_params.get('mod_index_fast', 0.5),
                    seed = driving_params.get('seed', 42)
                ) + mu
            else:
                noise = np.ones_like(t_array) * mu
                
            # Plot sample noise
            # plt.figure(figsize=(12, 4))
            # plt.plot(t_array, noise)
            # plt.title(f'Sample {drive} noise')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, f'sample_noise_{drive}.png'))
            # plt.close()
        
        # Run parameter sweep
        print(f"Running parameter sweep for {job_title}...")
        sweep_results = run_parameter_sweep(
            intrinsic_params=intrinsic_params,
            driving_params=driving_params,
            mu_p1_values=job_params['mu_p1_values'],
            mu_p2_values=job_params['mu_p2_values'],
            tmax=job_params['tmax'],
            dt=job_params['dt'],
            discard=job_params['discard']
        )
        
        # Save sweep results
        with open(os.path.join(output_dir, 'sweep_results.pkl'), 'wb') as f:
            pickle.dump(sweep_results, f)
        
        # Analyze couplings
        print("Analyzing couplings...")
        couplings = analyze_sweep_couplings(
            sweep_results,
            alpha_band=job_params['alpha_band'],
            gamma_band=job_params['gamma_band'],
            dt=job_params['dt']
        )
        
        # Plot coupling heatmaps
        plot_coupling_heatmaps(
            couplings,
            title=f"Couplings: {job_title}",
            save_path=os.path.join(output_dir, 'couplings.png')
        )
        
        # Analyze power
        print("Analyzing power...")
        power_results = analyze_sweep_power(
            sweep_results,
            alpha_band=job_params['alpha_band'],
            gamma_band=job_params['gamma_band'],
            dt=job_params['dt'],
            method='bandpass_hilbert'
        )
        
        # Plot power heatmaps
        plot_power_heatmaps_bottom_cbar(
            power_results,
            title=f"Band Power: {job_title}",
            log_scale=True,
            vmin_val=1e-2,
            vmax_val=1e1,
            save_path=os.path.join(output_dir, 'power.png')
        )
        freq_data = analyze_peak_frequencies(sweep_results)
        plot_frequency_heatmaps(freq_data, save_path=os.path.join(output_dir, 'frequency_heatmaps.png'))


        # Compute PEIX
        print("Computing PEIX...")
        peix_results = sweep_peix(sweep_results, params=intrinsic_params)
        
        # Plot PEIX heatmaps
        plot_peix_heatmaps(
            peix_results,
            title=f"PEIX: {job_title}",
            save_path=os.path.join(output_dir, 'peix.png')
        )
        
        # Save analysis results
        analysis_results = {
            'couplings': couplings,
            'power_results': power_results,
            'peix_results': peix_results,
            'freq_data': freq_data
        }
        with open(os.path.join(output_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(analysis_results, f)
        
        print(f"Job completed successfully. Results saved in {output_dir}")
        return {
            'output_dir': output_dir,
            'sweep_results': sweep_results,
            'analysis_results': analysis_results
        }
            

####### helper to summarize LaNMM driving parameters

from typing import Dict, Any, Optional

INJECTION_SYN = {"e1": 3, "e2": 8, "pv": 14}
VALID = {"constant", "am", "pulse", "multiscale"}

def _fmt_num(x: Optional[float], ndigits: int = 2) -> str:
    if x is None: return "None"
    try:
        xf = float(x); xi = int(round(xf))
        return str(xi) if abs(xf - xi) < 1e-9 else f"{xf:.{ndigits}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def _fmt_band(band, ndigits: int = 2) -> str:
    # If a variable like alpha_band was passed, show its name/value as-is
    if not isinstance(band, (list, tuple)): return str(band)
    if len(band) == 2: return f"{_fmt_num(band[0], ndigits)}–{_fmt_num(band[1], ndigits)} Hz"
    return str(band)

def _branch_strict(mode_raw: Any) -> str:
    """Exact simulator behavior (case-sensitive, no strip)."""
    if mode_raw == "multiscale": return "multiscale"
    if mode_raw == "am": return "am"
    if mode_raw == "pulse": return "pulse"
    return "constant"

def _branch_normalized(mode_raw: Any) -> str:
    """Normalization used for robust summary."""
    s = str(mode_raw).strip().lower() if isinstance(mode_raw, str) else str(mode_raw)
    return s if s in VALID else "constant"

def _ignored_for(branch: str) -> list:
    all_groups = ["am_params", "multiscale_params", "pulse_params"]
    keep = {
        "constant": set(),
        "am": {"am_params"},
        "pulse": {"pulse_params"},
        "multiscale": {"multiscale_params"},
    }.get(branch, set())
    return [g for g in all_groups if g not in keep]

def _am_rectification_risk(mu: float, amp: float, mfast: float, mslow: float) -> bool:
    # Conservative lower bound: mu - amp*(1+|mfast|)*(1+|mslow|) < 0 → possible clipping
    try: return (mu - amp * (1 + abs(mfast)) * (1 + abs(mslow))) < 0
    except Exception: return False

def summarize_lanmm_configuration(driving_params: Dict[str, Any],
                                  intrinsic_params: Optional[Dict[str, Any]] = None,
                                  ndigits: int = 2,
                                  markdown: bool = True,
                                  strict: bool = False,   # ← default robust
                                  debug: bool = False) -> str:
    """
    Summarize the effective config.
    - strict=False: normalize mode (strip/lower) for robustness; warn if strict branch would differ.
    - strict=True : mirror simulator’s exact branch (literal equality).
    """
    lines = []
    seed = driving_params.get("seed", None)

    lines.append(f"External drives injected at synapses: e1→{INJECTION_SYN['e1']}, e2→{INJECTION_SYN['e2']}, pv→{INJECTION_SYN['pv']}.")
    lines.append("Membrane potentials: vP1=u1+u2+u3+u11; vP2=u6+u7+u8+u12; vPV=u9+u10+u13+u14.")
    lines.append("Inputs are clipped to ≥ 0 before integration (rectification possible).")

    def add(s: str): lines.append(("- " + s) if markdown else s)

    for key in ("e1", "e2", "pv"):
        cfg = driving_params.get(key, {}) or {}
        mode_raw = cfg.get("mode", "constant")
        branch_exec = _branch_strict(mode_raw)
        branch_norm = _branch_normalized(mode_raw)
        branch = branch_exec if strict else branch_norm
        warn_norm = (branch_exec != branch_norm)

        mu = cfg.get("mu", 0.0)
        inj_syn = INJECTION_SYN[key]
        ignored = [g for g in _ignored_for(branch) if g in cfg]

        if branch == "constant":
            line = f"{key}: constant (μ = {_fmt_num(mu, ndigits)}), → syn {inj_syn}"
            notes = []
            if warn_norm: notes.append(f"mode='{mode_raw}' executes as '{branch_exec}', normalized='{branch_norm}'")
            if ignored:   notes.append(f"ignored: {', '.join(ignored)}")
            if notes: line += ". Note: " + "; ".join(notes)
            add(line + ".")
            continue

        if branch == "am":
            ap = cfg.get("am_params", {}) or {}
            cf   = ap.get("carrier_freq", 0.0)
            amp  = ap.get("carrier_amplitude", 0.0)
            eb   = ap.get("envelope_band", None)  # fast (m_fast)
            sb   = ap.get("slow_band", None)      # slow (m_slow)
            mslow = ap.get("mod_index_slow", 0.0)
            mfast = ap.get("mod_index_fast", 0.0)

            swapped = (key == "pv")  # your posted code swaps PV AM indices
            eff_mslow = (mfast if swapped else mslow)
            eff_mfast = (mslow if swapped else mfast)

            parts = [
                f"{key}: AM (μ = {_fmt_num(mu, ndigits)})",
                f"carrier {_fmt_num(cf, ndigits)} Hz",
                f"amplitude {_fmt_num(amp, ndigits)}",
                f"m_fast={_fmt_num(eff_mfast, ndigits)}, m_slow={_fmt_num(eff_mslow, ndigits)}",
                f"→ syn {inj_syn}",
            ]
            if eff_mfast not in (0, 0.0) and eb is not None:
                parts.append(f"fast band {_fmt_band(eb, ndigits)}")
            if eff_mslow not in (0, 0.0) and sb is not None:
                parts.append(f"slow band {_fmt_band(sb, ndigits)}")

            notes = []
            if warn_norm: notes.append(f"mode='{mode_raw}' executes as '{branch_exec}', normalized='{branch_norm}'")
            if cf == 0:   notes.append("carrier=0 → envelope-only AM (cos=1)")
            if amp == 0:  notes.append("amplitude=0 → AM suppressed (pure μ)")
            if swapped:   notes.append("PV AM indices are swapped in code (slow↔fast)")
            if _am_rectification_risk(mu, amp, eff_mfast, eff_mslow):
                notes.append("possible negative excursions pre-clip (rectification)")
            if ignored:   notes.append(f"ignored: {', '.join(ignored)}")

            line = "; ".join(parts) + "."
            if notes: line = line[:-1] + " Note: " + "; ".join(notes) + "."
            add(line)
            continue

        if branch == "pulse":
            pp = cfg.get("pulse_params", {}) or {}
            width = pp.get("pulse_width", None)
            height = pp.get("pulse_height", None)
            rate = pp.get("repetition_rate", None)
            notes = []
            if warn_norm: notes.append(f"mode='{mode_raw}' executes as '{branch_exec}', normalized='{branch_norm}'")
            if not height: notes.append("height=0 → pulses ineffective")
            if not width:  notes.append("width=0 → pulses ineffective")
            if ignored:    notes.append(f"ignored: {', '.join(ignored)}")
            line = (f"{key}: pulse (μ = {_fmt_num(mu, ndigits)}); width {_fmt_num(width, ndigits)} s; "
                    f"height {_fmt_num(height, ndigits)}; repetition {_fmt_num(rate, ndigits)} Hz; → syn {inj_syn}.")
            if notes: line = line[:-1] + " Note: " + "; ".join(notes) + "."
            add(line)
            continue

        if branch == "multiscale":
            ms = cfg.get("multiscale_params", {}) or {}
            sstd = ms.get("slow_std", None)
            salpha = ms.get("slow_alpha", None)
            fstd = ms.get("fast_std", None)
            fcut = ms.get("fast_cutoff", None)
            notes = []
            if warn_norm: notes.append(f"mode='{mode_raw}' executes as '{branch_exec}', normalized='{branch_norm}'")
            if (sstd in (0, None)) and (fstd in (0, None)): notes.append("slow/fast std=0 → no stochastic component")
            if ignored: notes.append(f"ignored: {', '.join(ignored)}")
            line = (f"{key}: multiscale (μ = {_fmt_num(mu, ndigits)}); slow std={_fmt_num(sstd, ndigits)}, "
                    f"α={_fmt_num(salpha, ndigits)}, fast std={_fmt_num(fstd, ndigits)}, cutoff={_fmt_num(fcut, ndigits)} Hz; → syn {inj_syn}.")
            if notes: line = line[:-1] + " Note: " + "; ".join(notes) + "."
            add(line)
            continue

    if seed is not None:
        add(f"Random seed: {_fmt_num(seed, ndigits)}")

    if intrinsic_params and intrinsic_params.get("include_PV_drive") is not None:
        add(f"include_PV_drive in intrinsic params = {intrinsic_params['include_PV_drive']} (not used in posted pipeline).")

    if debug:
        lines.append("--- debug ---")
        for key in ("e1","e2","pv"):
            mode_raw = (driving_params.get(key, {}) or {}).get("mode", None)
            lines.append(f"{key}.mode raw={mode_raw!r} | strict-branch={_branch_strict(mode_raw)} | normalized-branch={_branch_normalized(mode_raw)}")

    return "\n".join(lines)




# Example usage
if __name__ == "__main__":
    plot_synaptic_filter(A=3.25, a=100.0, 
                         tmax=0.15, nt=1000, 
                         fmax=50.0, nf=1000)
