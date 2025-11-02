import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt, hilbert
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import pearsonr
import concurrent.futures
import os
import json
from datetime import datetime
import pickle

###############################################################################
# 1) Intrinsic Model Parameters & Sigmoid
###############################################################################
def get_intrinsic_params():
    """
    Returns a dictionary of intrinsic LaNMM parameters.
    
    The parameters include:
      - Synaptic gains and time constants for AMPA, GABA_slow, and GABA_fast synapses.
      - Connectivity values (C_vals) for synapses 1 to 14.
        **Important:** Synapses 3, 8, and 14 are set to C=1 (they receive the external drive).
      - Synapse type identifiers.
      - Sigmoid parameters (thresholds, slope, maximum firing rate).
      - A flag 'include_PV_drive' which is True here, since we use 14 synapses.
    
    Returns
    -------
    dict
        Intrinsic model parameters.
    """
    params = {
        'A_AMPA': 3.25,
        'a_AMPA': 100.0,
        'A_GABA_slow': -22.0,
        'a_GABA_slow': 50.0,
        'A_GABA_fast': -30.0,
        'a_GABA_fast': 220.0,
        'C_vals': {
            1: 108.0,
            2: 33.7,
            3: 1.0,       # External drive for P1 is injected here.
            4: 135.0,
            5: 33.75,
            6: 70.0,
            7: 550.0,
            8: 1.0,       # External drive for P2 is injected here.
            9: 200.0,
            10: 100.0,
            11: 80.0,
            12: 200.0,
            13: 30.0,
            14: 1.0       # External drive for PV is injected here.
        },
        'syn_types': {
            1: 'AMPA',
            2: 'GABA_slow',
            3: 'AMPA',
            4: 'AMPA',
            5: 'AMPA',
            6: 'AMPA',
            7: 'GABA_fast',
            8: 'AMPA',
            9: 'AMPA',
            10: 'GABA_fast',
            11: 'AMPA',
            12: 'AMPA',
            13: 'AMPA',
            14: 'AMPA'
        },
        'v0_default': 6.0,
        'v0_p2': 1.0,
        'fmax': 5.0,
        'r_slope': 0.56,
        'include_PV_drive': True
    }
    return params


def get_intrinsic_params_extended(condition='healthy'):
    """
    Returns a dictionary of intrinsic LaNMM parameters with modifications based on the condition.
    
    Parameters
    ----------
    condition : str
        One of 'healthy', 'MCI', 'AD', 'Psychedelics', 'MCI+Psy', or 'AD+Psy'.
    
    The default values represent the healthy condition. The modifications are:
      - 'MCI': Set C_vals[7] to 300 (instead of 550).
      - 'AD': Set C_vals[7] to 140 (instead of 550).
      - 'Psychedelics': Multiply C_vals for synapses 4, 5, 12, and 13 by 1.3846153846.
      - 'MCI+Psy': Apply both the MCI and psychedelics modifications.
      - 'AD+Psy': Apply both the AD and psychedelics modifications.
    
    Returns
    -------
    dict
        Intrinsic model parameters.
    """
    params = {
        'A_AMPA': 3.25,
        'a_AMPA': 100.0,
        'A_GABA_slow': -22.0,
        'a_GABA_slow': 50.0,
        'A_GABA_fast': -30.0,
        'a_GABA_fast': 220.0,
        'C_vals': {
            1: 108.0,
            2: 33.7,
            3: 1.0,       # External drive for P1 is injected here.
            4: 135.0,
            5: 33.75,
            6: 70.0,
            7: 550.0,
            8: 1.0,       # External drive for P2 is injected here.
            9: 200.0,
            10: 100.0,
            11: 80.0,
            12: 200.0,
            13: 30.0,
            14: 1.0       # External drive for PV is injected here.
        },
        'syn_types': {
            1: 'AMPA',
            2: 'GABA_slow',
            3: 'AMPA',
            4: 'AMPA',
            5: 'AMPA',
            6: 'AMPA',
            7: 'GABA_fast',
            8: 'AMPA',
            9: 'AMPA',
            10: 'GABA_fast',
            11: 'AMPA',
            12: 'AMPA',
            13: 'AMPA',
            14: 'AMPA'
        },
        'v0_default': 6.0,
        'v0_p2': 1.0,
        'fmax': 5.0,
        'r_slope': 0.56,
        'include_PV_drive': True
    }
    
    # Define the psychedelic factor
    psychedelic_factor = 1.3846153846
    
    cond = condition.lower()
    if cond == 'mci':
        params['C_vals'][7] = 300.0
    elif cond == 'ad':
        params['C_vals'][7] = 140.0
    elif cond == 'psychedelics':
        #for syn in [4, 5, 12, 13]:
        for syn in [1, 3, 11]:    
            params['C_vals'][syn] *= psychedelic_factor
    elif cond == 'mci+psy':
        params['C_vals'][7] = 300.0
        #for syn in [4, 5, 12, 13]:
        for syn in [1, 3, 11]:
            params['C_vals'][syn] *= psychedelic_factor
    elif cond == 'ad+psy':
        params['C_vals'][7] = 140.0
        #for syn in [4, 5, 12, 13]:
        for syn in [1, 3, 11]:
            params['C_vals'][syn] *= psychedelic_factor
    # 'healthy' or any unrecognized condition defaults to no modification.
    
    return params

def sigmoid_pop(v, pop='default', params=None):
    """
    Computes the firing rate from membrane potential using a sigmoid function.
    
    Parameters
    ----------
    v : float or array-like
        Membrane potential(s).
    pop : str, optional
        'default' for most populations or 'P2' for the P2 population.
    params : dict, optional
        Intrinsic model parameters.
    
    Returns
    -------
    float or array-like
        Firing rate.
    """
    if params is None:
        params = get_intrinsic_params()
    v0 = params['v0_p2'] if pop == 'P2' else params['v0_default']
    fmax = params['fmax']
    r = params['r_slope']
    return fmax / (1.0 + np.exp(r * (v0 - v)))


def sigmoid_pop_linear(v, pop='default', params=None):
    """
    Linearized version of the sigmoid around its inflection point (v = v0).
    f(v) ≈ f(v0) + f'(v0) (v - v0)
    where f(v0) = fmax/2 and f'(v0) = fmax*r/4.
    """
    if params is None:
        params = get_intrinsic_params()
    v0 = params['v0_p2'] if pop == 'P2' else params['v0_default']
    fmax = params['fmax']
    r = params['r_slope']
    
    f_v0 = fmax / 2.0
    df_dv_v0 = fmax * r / 4.0
    
    return f_v0 + df_dv_v0 * (v - v0)

###############################################################################
# 2) Driving/Noise Parameters
###############################################################################
def get_driving_params():
    """
    Returns a dictionary of driving parameters for each external drive.
    
    For each drive ('e1', 'e2', 'pv'), you can specify:
      - 'mode': "constant", "multiscale", "am", or "pulse".
      - 'mu': the baseline (mean) value.
      - 'am_params': parameters for generating an AM signal.
      - 'multiscale_params': parameters for generating multiscale noise.
      - 'pulse_params': parameters for generating pulse signals.
    
    Also includes a global 'seed'.
    
    Returns
    -------
    dict
        Driving parameters.
    """
    params = {
        'e1': {
            'mode': 'constant',    # Options: "constant", "multiscale", "am", "pulse"
            'mu': 270.0,
            'am_params': {
                'carrier_freq': 0.0,
                'envelope_band': (8, 12),
                'slow_band': (0.5, 2),
                'carrier_amplitude': 400.0,
                'mod_index_slow': 0.9,
                'mod_index_fast': 0.5,
            },
            'multiscale_params': {
                'slow_std': 400.0,
                'slow_alpha': 0.99,
                'fast_std': 5.0,
                'fast_cutoff': 100.0,
            },
            'pulse_params': {
                'pulse_width': 0.05,  # 50ms pulses
                'pulse_height': 400.0,
                'repetition_rate': 10.0,  # 10 Hz
            }
        },
        'e2': {
            'mode': 'constant',    # Options: "constant", "multiscale", "am", "pulse"
            'mu': 90.0,
            'am_params': {
                'carrier_freq': 40.0,
                'envelope_band': (8, 12),
                'slow_band': (0.5, 2),
                'carrier_amplitude': 400.0,
                'mod_index_slow': 0.9,
                'mod_index_fast': 0.5,
            },
            'multiscale_params': {
                'slow_std': 400.0,
                'slow_alpha': 0.99,
                'fast_std': 5.0,
                'fast_cutoff': 100.0,
            },
            'pulse_params': {
                'pulse_width': 0.025,  # 25ms pulses
                'pulse_height': 400.0,
                'repetition_rate': 40.0,  # 40 Hz
            }
        },
        'pv': {
            'mode': 'constant',   # Options: "constant", "multiscale", "am", "pulse"
            'mu': 0.0,
            'am_params': {
                'carrier_freq': 0.0,
                'envelope_band': (8, 12),
                'slow_band': (0.5, 2),
                'carrier_amplitude': 400.0,
                'mod_index_slow': 0.9,
                'mod_index_fast': 0.5,
            },
            'multiscale_params': {
                'slow_std': 400.0,
                'slow_alpha': 0.99,
                'fast_std': 5.0,
                'fast_cutoff': 100.0,
            },
            'pulse_params': {
                'pulse_width': 0.01,  # 10ms pulses
                'pulse_height': 400.0,
                'repetition_rate': 20.0,  # 20 Hz
            }
        },
        'seed': 42
    }
    return params

def configure_driving_params(e1_config=None, e2_config=None, pv_config=None):
    """
    Configure driving parameters in a simple way, ensuring all parameters are properly set.
    
    Each config can be:
    - None: uses default constant drive
    - "constant": constant drive with default mu
    - "multiscale": multiscale noise with default parameters
    - "pulse": pulse signal with default parameters
    - A dictionary with detailed configuration, e.g.:
        {'mode': 'am',
         'mu': 270.0,
         'carrier_freq': 10.0,  # only for 'am' mode
         'carrier_amplitude': 400.0,  # only for 'am' mode
         'mod_index_slow': 0.9,  # only for 'am' mode
         'mod_index_fast': 0.5}  # only for 'am' mode
        or:
        {'mode': 'pulse',
         'mu': 270.0,
         'pulse_width': 0.05,  # only for 'pulse' mode
         'pulse_height': 400.0,  # only for 'pulse' mode
         'repetition_rate': 10.0}  # only for 'pulse' mode
    
    Parameters
    ----------
    e1_config : str or dict, optional
        Configuration for e1 (P1) drive
    e2_config : str or dict, optional
        Configuration for e2 (P2) drive
    pv_config : str or dict, optional
        Configuration for PV drive
    
    Returns
    -------
    dict
        Complete driving parameters dictionary
    """
    # Get default parameters to ensure complete structure
    params = get_driving_params()
    
    # Helper function to update a single drive configuration
    def update_drive_config(drive_params, config):
        if config is None:
            drive_params['mode'] = 'constant'
        elif isinstance(config, str):
            drive_params['mode'] = config
        elif isinstance(config, dict):
            # Update mode
            if 'mode' in config:
                drive_params['mode'] = config['mode']
            
            # Update mu if provided
            if 'mu' in config:
                drive_params['mu'] = config['mu']
            
            # Update AM parameters if provided and mode is 'am'
            if drive_params['mode'] == 'am':
                if 'carrier_freq' in config:
                    drive_params['am_params']['carrier_freq'] = config['carrier_freq']
                if 'carrier_amplitude' in config:
                    drive_params['am_params']['carrier_amplitude'] = config['carrier_amplitude']
                if 'mod_index_slow' in config:
                    drive_params['am_params']['mod_index_slow'] = config['mod_index_slow']
                if 'mod_index_fast' in config:
                    drive_params['am_params']['mod_index_fast'] = config['mod_index_fast']
                if 'envelope_band' in config:
                    drive_params['am_params']['envelope_band'] = config['envelope_band']
                if 'slow_band' in config:
                    drive_params['am_params']['slow_band'] = config['slow_band']
            
            # Update pulse parameters if provided and mode is 'pulse'
            if drive_params['mode'] == 'pulse':
                if 'pulse_width' in config:
                    drive_params['pulse_params']['pulse_width'] = config['pulse_width']
                if 'pulse_height' in config:
                    drive_params['pulse_params']['pulse_height'] = config['pulse_height']
                if 'repetition_rate' in config:
                    drive_params['pulse_params']['repetition_rate'] = config['repetition_rate']
            
            # Update multiscale parameters if provided and mode is 'multiscale'
            if drive_params['mode'] == 'multiscale':
                if 'slow_std' in config:
                    drive_params['multiscale_params']['slow_std'] = config['slow_std']
                if 'slow_alpha' in config:
                    drive_params['multiscale_params']['slow_alpha'] = config['slow_alpha']
                if 'fast_std' in config:
                    drive_params['multiscale_params']['fast_std'] = config['fast_std']
                if 'fast_cutoff' in config:
                    drive_params['multiscale_params']['fast_cutoff'] = config['fast_cutoff']
    
    # Update configurations for each drive while maintaining complete structure
    update_drive_config(params['e1'], e1_config)
    update_drive_config(params['e2'], e2_config)
    update_drive_config(params['pv'], pv_config)
    
    return params

###############################################################################
# 3) Signal Generation Functions
###############################################################################
def generate_multiscale_noise(t_array, base_mean, seed, fs,
                              slow_std=400.0, slow_alpha=0.99,
                              fast_std=5.0, fast_cutoff=100.0, floor_val=1e-4):
    """
    Generates a noise signal by combining a slow AR(1) drift with fast noise.
    
    Parameters
    ----------
    t_array : array-like
        Time vector.
    base_mean : float
        Baseline mean.
    seed : int
        Random seed.
    fs : float
        Sampling frequency.
    slow_std, slow_alpha, fast_std, fast_cutoff : float
        Noise parameters.
    floor_val : float
        Minimum output value.
    
    Returns
    -------
    np.array
        Generated noise signal.
    """
    rng = np.random.RandomState(seed)
    N = len(t_array)
    dt_val = t_array[1] - t_array[0]
    slow_part = np.zeros(N)
    fast_part = np.zeros(N)
    slow_part[0] = base_mean
    for i in range(1, N):
        step_slow = rng.randn() * slow_std
        slow_part[i] = slow_alpha * slow_part[i-1] + (1.0 - slow_alpha) * (base_mean + step_slow)
    fast_white = rng.randn(N) * fast_std
    nyq = 0.5 * fs
    b, a = butter(4, fast_cutoff/nyq, btype='low')
    fast_part = filtfilt(b, a, fast_white)
    out_noise = slow_part + fast_part
    return np.clip(out_noise, floor_val, None)

def bandpass_filter_am(signal, t_array, band, order=4):
    """
    Bandpass filters a signal using a Butterworth filter in SOS form.
    
    Parameters
    ----------
    signal : array-like
        Input signal.
    t_array : array-like
        Time vector.
    band : tuple
        (low, high) cutoff frequencies in Hz.
    order : int, optional
        Filter order.
    
    Returns
    -------
    np.array
        Filtered signal.
    """
    dt = t_array[1] - t_array[0]
    fs = 1.0 / dt
    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def generate_nested_am_signal(t_array, carrier_freq, envelope_band, slow_band,
                              carrier_amplitude, mod_index_slow, mod_index_fast, seed):
    """
    Generates a nested AM signal:
    
         S(t) = E(t) * cos(2π f_c t)
    
    where the envelope is given by:
    
         E(t) = carrier_amplitude * [1 + mod_index_slow*m1(t)] * [1 + mod_index_fast*m2(t)]
    
    m1(t) and m2(t) are obtained by bandpass filtering white noise.
    
    Parameters
    ----------
    t_array : array-like
        Time vector.
    carrier_freq : float
        Carrier frequency in Hz.
    envelope_band : tuple
        Frequency band (Hz) for primary modulator.
    slow_band : tuple
        Frequency band (Hz) for slow modulator.
    carrier_amplitude : float
        Scaling factor for envelope.
    mod_index_slow, mod_index_fast : float
        Modulation indices.
    seed : int
        Random seed.
    
    Returns
    -------
    np.array
        Generated AM signal (with DC removed).
    """
    rng = np.random.RandomState(seed)
    N = len(t_array)
    # Primary modulator m1
    noise1 = rng.randn(N)
    m1 = bandpass_filter_am(noise1, t_array, envelope_band)
    m1_range = m1.max() - m1.min()
    m1_norm = 2 * (m1 - m1.min()) / (m1_range + 1e-8) - 1 if m1_range != 0 else np.zeros_like(m1)
    # Slow modulator m2
    noise2 = rng.randn(N)
    m2 = bandpass_filter_am(noise2, t_array, slow_band)
    m2_range = m2.max() - m2.min()
    m2_norm = 2 * (m2 - m2.min()) / (m2_range + 1e-8) - 1 if m2_range != 0 else np.zeros_like(m2)
    envelope = carrier_amplitude * (1 + mod_index_fast * m1_norm) * (1 + mod_index_slow * m2_norm)
    carrier = np.cos(2 * np.pi * carrier_freq * t_array)
    S = envelope * carrier
    return S - np.mean(S)

def generate_pulse_signal(t_array, pulse_width, pulse_height, repetition_rate, baseline=0.0):
    """
    Generates a periodic pulse signal.
    
    Parameters
    ----------
    t_array : array-like
        Time vector.
    pulse_width : float
        Width of each pulse in seconds.
    pulse_height : float
        Height/amplitude of the pulse above baseline.
    repetition_rate : float
        Number of pulses per second (Hz).
    baseline : float, optional
        Baseline value (default 0.0).
    
    Returns
    -------
    np.array
        Generated pulse signal.
    """
    period = 1.0 / repetition_rate
    signal = np.ones_like(t_array) * baseline
    
    # Generate pulses
    for t_start in np.arange(0, t_array[-1], period):
        mask = (t_array >= t_start) & (t_array < t_start + pulse_width)
        signal[mask] = baseline + pulse_height
    
    return signal

###############################################################################
# 4) Unified ODE Function with 14 Synapses (Mapping as Specified)
###############################################################################
def lanmm_ode_unified(Y, t, params, e1_array, e2_array, pv_array, t_array):
    """
    Unified ODE function for the LaNMM model with 14 synapses (28 state variables).
    
    External drives are injected only into:
      - Synapse 3 (for P1 drive, using external signal φ_e1),
      - Synapse 8 (for P2 drive, using external signal φ_e2),
      - Synapse 14 (for PV drive, using external signal φ_pv).
    
    The mapping for membrane potentials is:
        vP1  = u(1) + u(2) + u(3) + u(11)
        vSS  = u(4)
        vSST = u(5)
        vP2  = u(6) + u(7) + u(8) + u(12)
        vPV  = u(9) + u(10) + u(13) + u(14)
    
    Firing rates are computed as:
        sP1  = sigmoid_pop(vP1, 'default', params)
        sSS  = sigmoid_pop(vSS, 'default', params)
        sSST = sigmoid_pop(vSST, 'default', params)
        sP2  = sigmoid_pop(vP2, 'P2', params)
        sPV  = sigmoid_pop(vPV, 'default', params)
    
    Synaptic dynamics are given by:
        du_dt = z(s)
        dz_dt = a * A * (C_s * presyn_rate) - 2 * a * z(s) - a^2 * u(s)
    
    The external drive is injected via interpolation from the corresponding array.
    
    Parameters
    ----------
    Y : array-like
        State vector (length 28).
    t : float
        Current time.
    params : dict
        Intrinsic model parameters.
    e1_array : array-like
        External input array for P1.
    e2_array : array-like
        External input array for P2.
    pv_array : array-like
        External input array for PV.
    t_array : array-like
        Time vector for external inputs.
    
    Returns
    -------
    list
        Derivative of Y.
    """
    # Helper functions: state variable access.
    u = lambda s: Y[2*(s-1)]
    z = lambda s: Y[2*(s-1) + 1]
    
    # Interpolation function for external signals.
    def interp(signal):
        idx = np.searchsorted(t_array, t)
        if idx <= 0:
            return signal[0]
        elif idx >= len(t_array):
            return signal[-1]
        else:
            t0, t1 = t_array[idx-1], t_array[idx]
            w = (t - t0) / (t1 - t0)
            return (1 - w) * signal[idx-1] + w * signal[idx]
    
    # Interpolate each external drive.
    phi_e1 = interp(e1_array)
    phi_e2 = interp(e2_array)
    phi_pv = interp(pv_array)
    
    # Compute membrane potentials.
    vP1 = u(1) + u(2) + u(3) + u(11)
    vSS = u(4)
    vSST = u(5)
    vP2 = u(6) + u(7) + u(8) + u(12)
    vPV = u(9) + u(10) + u(13) + u(14)
    
    
    # Compute firing rates.
    sP1 = sigmoid_pop(vP1, pop='default', params=params)
    sSS = sigmoid_pop(vSS, pop='default', params=params)
    sSST = sigmoid_pop(vSST, pop='default', params=params)
    sP2 = sigmoid_pop(vP2, pop='P2', params=params)
    sPV = sigmoid_pop(vPV, pop='default', params=params)
    
    # Synaptic ODE function.
    def syn_ode(s, presyn_rate):
        stype = params['syn_types'][s]
        C_s = params['C_vals'][s]
        if stype == 'AMPA':
            a_val = params['a_AMPA']
            A_val = params['A_AMPA']
        elif stype == 'GABA_slow':
            a_val = params['a_GABA_slow']
            A_val = params['A_GABA_slow']
        elif stype == 'GABA_fast':
            a_val = params['a_GABA_fast']
            A_val = params['A_GABA_fast']
        else:
            raise ValueError("Unknown synapse type")
        du_dt = z(s)
        dz_dt = a_val * A_val * (C_s * presyn_rate) - 2 * a_val * z(s) - (a_val**2) * u(s)
        return du_dt, dz_dt
    
    # Assign synaptic drives following the mapping:
    du1, dz1   = syn_ode(1, sSS)
    du2, dz2   = syn_ode(2, sSST)
    du3, dz3   = syn_ode(3, phi_e1)     # External drive for P1 is injected here.
    du4, dz4   = syn_ode(4, sP1)
    du5, dz5   = syn_ode(5, sP1)
    du6, dz6   = syn_ode(6, sP2)
    du7, dz7   = syn_ode(7, sPV)
    du8, dz8   = syn_ode(8, phi_e2)     # External drive for P2 is injected here.
    du9, dz9   = syn_ode(9, sP2)
    du10, dz10 = syn_ode(10, sPV)
    du11, dz11 = syn_ode(11, sP2)
    du12, dz12 = syn_ode(12, sP1)
    du13, dz13 = syn_ode(13, sP1)
    du14, dz14 = syn_ode(14, phi_pv)     # External drive for PV is injected here.
    
    return [du1, dz1, du2, dz2, du3, dz3, du4, dz4,
            du5, dz5, du6, dz6, du7, dz7, du8, dz8,
            du9, dz9, du10, dz10, du11, dz11, du12, dz12,
            du13, dz13, du14, dz14]

###############################################################################
# 5) Unified Simulation Wrapper
###############################################################################
def run_unified_simulation(intrinsic_params, driving_params, tmax=4.0, dt=0.001, discard=1.0):
    """
    Runs a unified LaNMM simulation using the unified ODE function with 14 synapses.
    
    The driving_params dictionary must include separate configurations for each drive:
       - 'e1': settings for the P1 drive.
       - 'e2': settings for the P2 drive.
       - 'pv': settings for the PV drive.
    Each drive's settings include:
       - 'mode': "constant", "multiscale", "am", or "pulse"
       - 'mu': the baseline mean value.
       - 'am_params', 'multiscale_params', or 'pulse_params' as needed.
    
    Returns
    -------
    dict
        Dictionary containing:
          't': time vector (after discarding transient),
          'vP1': membrane potential for P1,
          'vP2': membrane potential for P2,
          'vPV': membrane potential for PV,
          'e1_array', 'e2_array', 'pv_array': the external input arrays.
    """
    t_array = np.arange(0, tmax, dt)
    fs = 1.0 / dt
    seed = driving_params.get("seed", 42)
    
    # --- Generate e1 drive ---
    e1_cfg = driving_params.get("e1", {})
    mode_e1 = e1_cfg.get("mode", "constant")
    mu_e1 = e1_cfg.get("mu", 200.0)
    if mode_e1 == "multiscale":
        e1_array = generate_multiscale_noise(
            t_array, base_mean=mu_e1, seed=seed, fs=fs,
            **e1_cfg.get("multiscale_params", {}))
    elif mode_e1 == "am":
        am_params = e1_cfg.get("am_params", {})
        e1_array = generate_nested_am_signal(
            t_array,
            carrier_freq=am_params.get("carrier_freq", 10.0),
            envelope_band=am_params.get("envelope_band", (8,12)),
            slow_band=am_params.get("slow_band", (0.5,2)),
            carrier_amplitude=am_params.get("carrier_amplitude", 1.0),
            mod_index_slow=am_params.get("mod_index_slow", 0.5),
            mod_index_fast=am_params.get("mod_index_fast", 0.5),
            seed=seed
        ) + mu_e1
    elif mode_e1 == "pulse":
        pulse_params = e1_cfg.get("pulse_params", {})
        e1_array = generate_pulse_signal(
            t_array,
            pulse_width=pulse_params.get("pulse_width", 0.05),
            pulse_height=pulse_params.get("pulse_height", 400.0),
            repetition_rate=pulse_params.get("repetition_rate", 10.0),
            baseline=mu_e1
        )
    else:
        e1_array = np.ones_like(t_array) * mu_e1

    # --- Generate e2 drive ---
    e2_cfg = driving_params.get("e2", {})
    mode_e2 = e2_cfg.get("mode", "constant")
    mu_e2 = e2_cfg.get("mu", 90.0)
    if mode_e2 == "multiscale":
        e2_array = generate_multiscale_noise(
            t_array, base_mean=mu_e2, seed=seed+1, fs=fs,
            **e2_cfg.get("multiscale_params", {}))
    elif mode_e2 == "am":
        am_params = e2_cfg.get("am_params", {})
        e2_array = generate_nested_am_signal(
            t_array,
            carrier_freq=am_params.get("carrier_freq", 40.0),
            envelope_band=am_params.get("envelope_band", (8,12)),
            slow_band=am_params.get("slow_band", (0.5,2)),
            carrier_amplitude=am_params.get("carrier_amplitude", 1.0),
            mod_index_slow=am_params.get("mod_index_slow", 0.5),
            mod_index_fast=am_params.get("mod_index_fast", 0.5),
            seed=seed+1
        ) + mu_e2
    elif mode_e2 == "pulse":
        pulse_params = e2_cfg.get("pulse_params", {})
        e2_array = generate_pulse_signal(
            t_array,
            pulse_width=pulse_params.get("pulse_width", 0.025),
            pulse_height=pulse_params.get("pulse_height", 400.0),
            repetition_rate=pulse_params.get("repetition_rate", 40.0),
            baseline=mu_e2
        )
    else:
        e2_array = np.ones_like(t_array) * mu_e2

    # --- Generate pv drive ---
    pv_cfg = driving_params.get("pv", {})
    mode_pv = pv_cfg.get("mode", "constant")
    mu_pv = pv_cfg.get("mu", 0.0)
    if mode_pv == "multiscale":
        pv_array = generate_multiscale_noise(
            t_array, base_mean=mu_pv, seed=seed+2, fs=fs,
            **pv_cfg.get("multiscale_params", {}))
    elif mode_pv == "am":
        am_params = pv_cfg.get("am_params", {})
        pv_array = generate_nested_am_signal(
            t_array,
            carrier_freq=am_params.get("carrier_freq", 0.0),
            envelope_band=am_params.get("envelope_band", (8,12)),
            slow_band=am_params.get("slow_band", (0.5,2)),
            carrier_amplitude=am_params.get("carrier_amplitude", 1.0),
            mod_index_slow=am_params.get("mod_index_slow", 0.5),
            mod_index_fast=am_params.get("mod_index_fast", 0.5),
            seed=seed+2
        ) + mu_pv
    elif mode_pv == "pulse":
        pulse_params = pv_cfg.get("pulse_params", {})
        pv_array = generate_pulse_signal(
            t_array,
            pulse_width=pulse_params.get("pulse_width", 0.01),
            pulse_height=pulse_params.get("pulse_height", 400.0),
            repetition_rate=pulse_params.get("repetition_rate", 20.0),
            baseline=mu_pv
        )
    else:
        pv_array = np.ones_like(t_array) * mu_pv

    # clip the e1_array, e2_array, and pv_array to be greater than 0
    e1_array = np.clip(e1_array, 0, None)
    e2_array = np.clip(e2_array, 0, None)
    pv_array = np.clip(pv_array, 0, None)

    # --- Call unified ODE function ---
    Y0 = np.zeros(28)
    def ode_wrap(Y, t):
        return lanmm_ode_unified(Y, t, intrinsic_params, e1_array, e2_array, pv_array, t_array)
    #sol = odeint(ode_wrap, Y0, t_array, mxstep=20000)


    # compute solution with RK4
    sol = solve_ivp(lambda t, y: lanmm_ode_unified(y, t, intrinsic_params, 
                                                      e1_array, e2_array, pv_array, t_array),
                       (t_array[0], t_array[-1]), Y0, 
                       method='RK45',
                       t_eval=t_array,
                       rtol=1e-8, atol=1e-8, max_step=10000)
    sol = sol.y.T
    
    # --- Extract membrane potentials ---
    def u(s):
        return sol[:, 2*(s-1)]
    vP1 = u(1) + u(2) + u(3) + u(11)
    vP2 = u(6) + u(7) + u(8) + u(12)
    vPV = u(9) + u(10) + u(13) + u(14)
    
    mask = (t_array >= discard)
    results = {
        't': t_array[mask],
        'vP1': vP1[mask],
        'vP2': vP2[mask],
        'vPV': vPV[mask],
        'e1_array': e1_array[mask],
        'e2_array': e2_array[mask],
        'pv_array': pv_array[mask], 
        'u1': u(1)[mask],
        'u2': u(2)[mask],
        'u3': u(3)[mask],
        'u4': u(4)[mask],
        'u5': u(5)[mask],
        'u6': u(6)[mask],
        'u7': u(7)[mask],
        'u8': u(8)[mask],
        'u9': u(9)[mask],
        'u10': u(10)[mask],
        'u11': u(11)[mask],
        'u12': u(12)[mask],
        'u13': u(13)[mask],
        'u14': u(14)[mask]
    }
    return results


import copy
import concurrent.futures
from tqdm.notebook import tqdm  # Use tqdm.notebook in Jupyter, or tqdm in a script

def simulate_for_pair(args):
    """
    Worker function for a single parameter pair simulation.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (m1, m2, intrinsic_params, driving_params, tmax, dt, discard).
    
    Returns
    -------
    tuple
        ((m1, m2), simulation_data)
    """
    m1, m2, intrinsic_params, driving_params, tmax, dt, discard = args
    # Create a deep copy so that each simulation uses its own driving parameters.
    drive_cfg = copy.deepcopy(driving_params)
    drive_cfg["e1"]["mu"] = m1
    drive_cfg["e2"]["mu"] = m2
    sim_data = run_unified_simulation(intrinsic_params, drive_cfg, tmax=tmax, dt=dt, discard=discard)
    return (m1, m2), sim_data

def run_parameter_sweep(intrinsic_params, driving_params, mu_p1_values, mu_p2_values,
                        tmax=4.0, dt=0.001, discard=1.0):
    """
    Performs a parameter sweep over mu values for e1 and e2 drives.
    
    Parameters
    ----------
    intrinsic_params : dict
        Intrinsic (biophysical) model parameters.
    driving_params : dict
        Driving parameters for external inputs (contains separate entries for 'e1', 'e2', and 'pv').
    mu_p1_values : iterable
        Iterable of baseline values for the e1 drive.
    mu_p2_values : iterable
        Iterable of baseline values for the e2 drive.
    tmax : float, optional
        Maximum simulation time.
    dt : float, optional
        Time step.
    discard : float, optional
        Transient time to discard.
    
    Returns
    -------
    dict
        Dictionary with keys (m1, m2) mapping to simulation result dictionaries.
    """
    all_args = [
        (m1, m2, intrinsic_params, driving_params, tmax, dt, discard)
        for m1 in mu_p1_values for m2 in mu_p2_values
    ]
    
    all_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(simulate_for_pair, arg) for arg in all_args]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Parameter sweep progress"):
            (m1, m2), sim_data = future.result()
            all_results[(m1, m2)] = sim_data
    return all_results

###############################################################################
# 6) Frequency Analysis Functions
###############################################################################
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
            vP2 = result['vP2']  # Using vP2 for analysis
            
            alpha_peaks[i, j] = compute_peak_frequency(vP2, fs, alpha_band)
            gamma_peaks[i, j] = compute_peak_frequency(vP2, fs, gamma_band)
    
    return {
        'alpha_peaks': alpha_peaks,
        'gamma_peaks': gamma_peaks,
        'm1_values': m1_values,
        'm2_values': m2_values
    }
import matplotlib.ticker as ticker

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
    
    # Plot alpha peaks
    im1 = ax1.imshow(freq_data['alpha_peaks'].T, origin='lower', aspect='auto',
                     extent=[min(freq_data['m1_values']), max(freq_data['m1_values']),
                            min(freq_data['m2_values']), max(freq_data['m2_values'])])
    ax1.set_xlabel('P1 drive (Hz)')
    ax1.set_ylabel('P2 drive (Hz)')
    ax1.set_title('Peak Alpha Frequency (Hz)')
    # ensure colorbar ticks are integers
    cbar = ax1.collections[0].colorbar
    cbar.set_ticks(np.arange(cbar.vmin, cbar.vmax + 1, 1))
    plt.colorbar(im1, ax=ax1)
    
    # Plot gamma peaks
    im2 = ax2.imshow(freq_data['gamma_peaks'].T, origin='lower', aspect='auto',
                     extent=[min(freq_data['m1_values']), max(freq_data['m1_values']),
                            min(freq_data['m2_values']), max(freq_data['m2_values'])])
    ax2.set_xlabel('P1 drive (Hz)')
    ax2.set_ylabel('P2 drive (Hz)')
    ax2.set_title('Peak Gamma Frequency (Hz)')
    plt.colorbar(im2, ax=ax2)
    # ensure colorbar ticks are integers    
    cbar = ax2.collections[0].colorbar
    #cbar.set_ticks(np.arange(cbar.vmin, cbar.vmax + 1, 1))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

###############################################################################
# 7) Example Usage and Plotting
###############################################################################
if __name__ == "__main__":
    # Retrieve intrinsic and driving parameters.
    intrinsic_params = get_intrinsic_params()
    driving_params = get_driving_params()
    
    # Customize each drive independently:
    # e1: AM drive at 10 Hz.
    driving_params["e1"]["mode"] = "am"
    driving_params["e1"]["mu"] = 270.0
    driving_params["e1"]["am_params"]["carrier_freq"] = 10.0
    
    # e2: AM drive at 40 Hz.
    driving_params["e2"]["mode"] = "am"
    driving_params["e2"]["mu"] = 90.0
    driving_params["e2"]["am_params"]["carrier_freq"] = 40.0
    
    # pv: constant drive.
    driving_params["pv"]["mode"] = "constant"
    driving_params["pv"]["mu"] = 0.0
    
    # Run unified simulation.
    sim_results = run_unified_simulation(intrinsic_params, driving_params,
                                         tmax=4.0, dt=0.001, discard=1.0)
    
    # Plot membrane potentials.
    plt.figure(figsize=(18,6))
    plt.plot(sim_results['t'], sim_results['vP1'], label="vP1")
    plt.plot(sim_results['t'], sim_results['vP2'], label="vP2")
    plt.plot(sim_results['t'], sim_results['vPV'], label="vPV")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential")
    plt.title("LaNMM Simulation Results (MP)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot inputs
    plt.figure(figsize=(18,6))
    plt.plot(sim_results['t'], sim_results['e1_array'], label="P1 input")
    plt.plot(sim_results['t'], sim_results['e2_array'], label="P2 input")
    plt.plot(sim_results['t'], sim_results['pv_array'], label="PV input")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Inputs")
    plt.legend()
    plt.grid(True)
    plt.show()    


    # Optionally, you can customize additional parts of driving_params here.

    mu_p1_values = range(50, 400, 50) # Example range for e1 baseline.
    mu_p2_values = range(50, 400, 50)    # Example range for e2 baseline.

    sweep_results = run_parameter_sweep(intrinsic_params, driving_params,
                                        mu_p1_values, mu_p2_values,
                                        tmax=4.0, dt=0.001, discard=1.0)

    # Example: plot vP1 for a specific (m1, m2) pair:
    key = (270, 90)
    result = sweep_results[key]
    plt.figure(figsize=(12, 6))
    plt.plot(result['t'], result['vP1'], label='vP1')
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential")
    plt.title(f"Simulation for e1 mu={key[0]}, e2 mu={key[1]}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Analyze peak frequencies from the sweep results
    freq_data = analyze_peak_frequencies(sweep_results)
    
    # Plot frequency heatmaps
    plot_frequency_heatmaps(freq_data)


    # demo how to generate a multiscale noise signal
    t_array = np.linspace(0, 4.0, 10000)
    noise = generate_multiscale_noise(t_array, 0.0, 1, 1000, slow_std=400.0, slow_alpha=0.99, fast_std=5.0, fast_cutoff=100.0)
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, noise)
    plt.show()  

    # demo how to generate a pulse signal
    pulse = generate_pulse_signal(t_array, 0.05, 400.0, 10.0)
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, pulse)
    plt.show()  

    # demo how to generate a AM signal
    am = generate_nested_am_signal(t_array, 10.0, (8, 12), (0.5, 2), 400.0, 0.9, 0.5, 1)
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, am)
    plt.show()    

    am = generate_nested_am_signal(t_array, 30.0, (8, 12), (0.5, 2), 400.0, 0.9, 0.5, 1)
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, am, color = 'red')
    plt.show()
    # add envelope to the am signal
    envelope = np.abs(hilbert(am))
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, envelope)
    plt.show()

    # demo how to generate a multiscale noise signal
    noise = generate_multiscale_noise(t_array, 0.0, 1, 1000, slow_std=400.0, slow_alpha=0.99, fast_std=5.0, fast_cutoff=100.0)
    plt.figure(figsize=(12, 6))
    plt.plot(t_array, noise)
    plt.show()  
    #remove y axis label and ticks
    plt.gca().set_ylabel('')
    plt.gca().set_yticks([])
    plt.show()  

    # now plot the am signal divided by the envelope
    #     
    
    