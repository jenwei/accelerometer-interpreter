import numpy as np
import thinkdsp
from hmmlearn.hmm import GaussianHMM

from utils import evenly_sample_ts
from magnitude import magnitude
from dominant_frequency import find_dominant


def preprocess(df):
    """
    Arguments
    ---------
    df: a Dataframe
        columns x, y, z, and time
    
    Returns
    -------
    interp_vals: array-like, shape (n_samples,)
	   evenly spaced in time values, (i.e. magnitude of the accleration vector)	
    """
    
    # Calculate Magnitude
    vals = magnitude(df[["x", "y", "z"]].values)

    # Interpolate
    time = df.time.values
    evenly_sampled_ts = evenly_sample_ts(time)
    interp_vals = np.interp(evenly_sampled_ts, time, vals) 

    return interp_vals

def extract_features(
    vals,
    start0=0,
    window_size=30,
    step_size=15,
    n_windows=20,
    dom_freq_method="autocorr"):
    """
    Arguments
    ---------
    vals: array-like, shape (n_samples,)
        evenly spaced in time values, (i.e. magnitude of the accleration vector)
    
    start0: number
        time to start extracting window segments

    window_size: number
        how many indices the windows should be

    step_size: number
        how many indices each step should be.  convolution would be step_size=1

    n_windows: integer
        number of segments to extract. Equal to the length of the sequence of
        observations, obs.

    dom_freq_method: string, default, "autocorr"
        available strings:
        "autocorr" - calculated using autocorrelation
        "spectrum" - calculated using peaks method of frequency spectrum

    Returns
    -------
    obs: array-like, shape (n_windows, n_features)
        current features: dom_amp, dom_freq
    """
    dom_amp = np.zeros(n_windows)
    dom_freq = np.zeros(n_windows)

    # Calculate Waves
    wave = thinkdsp.Wave(vals, framerate=1)

    for i in range(n_windows):
        seg = wave.segment(start=start0+i*step_size, duration=window_size)

        # Unbiasing the Wave to make Spectrum amplitude at frequency 0, equal to 0
        seg.unbias()

        # Convert to Frequency Domain
        spectrum = seg.make_spectrum()

        # Extract Peaks
        # peak[0] the highest peak
        peaks = spectrum.peaks()


        # peak[0][0] is the amplitude of the highest peak
        # peak[0][1] is the frequency of the highest peak
        dom_amp[i] = peaks[0][0]
        
        if dom_freq_method == "autocorr":
            dom_freq[i] = find_dominant(seg)
        elif dom_freq_method == "spectrum":
            dom_freq[i] = peaks[0][1]
        else:
            raise ValueError(
                "dom_freq_method argment must be 'autocorr' or 'spectrum'"
            )

    return np.column_stack((dom_freq, dom_amp))