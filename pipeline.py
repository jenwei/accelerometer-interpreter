import numpy as np
import thinkdsp
from hmmlearn.hmm import GaussianHMM
from collections import deque

from utils import evenly_sample_ts, vStackMatrices
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

def extract_features_with_sliding_window(
        vals,
        start0=0,
        window_size=30,
        step_size=15,
        n_windows=20
    ):
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

    Returns
    -------
    obs : list
        List of array-like observation sequences, each of which has shape 
        (n_i, n_features), where n_i is the length of the i_th observation.
    """
    dom_amp = np.zeros(n_windows)
    dom_freq = np.zeros(n_windows)

    # Calculate Waves
    wave = thinkdsp.Wave(vals, framerate=1)

    # (n - w) / s + 1
    n_total_windows = int((wave.duration - window_size) / step_size) + 1
    
    # Sliding Windowing, With the Deque thingy
    # Every n_windows, 
    
    # Initialize DQ
    first_n_windows = [extract_features(wave.segment(start=start0+i*step_size, duration=window_size)) for i in range(n_windows)] 
    dq = deque(first_n_windows)

    # Initialize and Extract first observation sequence
    obs = [np.vstack(list(dq))]

    # Slide the bigger window of segments to extract more observation seequences
    for i in range(n_windows, n_total_windows):
        _ = dq.popleft()
        dq.append(
            extract_features(
                wave.segment(start=start0+i*step_size, duration=window_size)
            )
        )
        obs.append(np.vstack(list(dq)))

    return obs

def extract_features(seg, do_hamming=True, dom_freq_method="autocorr"):
    """
    Arguments
    ---------
    seg: a wave segment (or window) to apply feature extraction

    do_hamming: boolean, default=True
        whether to apply a hamming window to each segment before FFT.

    dom_freq_method: string, default, "autocorr"
        available strings:
        "autocorr" - calculated using autocorrelation
        "spectrum" - calculated using peaks method of frequency spectrum

    Returns
    -------
    features: array-like, length n_features
    """
    # Unbiasing the Wave to make Spectrum amplitude at frequency 0, equal to 0
    seg.unbias()

    if do_hamming:
        seg.hamming()

    # Convert to Frequency Domain
    spectrum = seg.make_spectrum()

    # Extract Peaks
    # peak[0] the highest peak
    peaks = spectrum.peaks()

    # peak[0][0] is the amplitude of the highest peak
    # peak[0][1] is the frequency of the highest peak
    dom_amp = peaks[0][0]
    
    if dom_freq_method == "autocorr":
        dom_freq = find_dominant(seg)
    elif dom_freq_method == "spectrum":
        dom_freq = peaks[0][1]
    else:
        raise ValueError(
            "dom_freq_method argment must be 'autocorr' or 'spectrum'"
        )

    return [dom_freq, dom_amp]  

def learn(feature_dict):
    """
    Arguments
    ---------
    feature_dict: a dictionary of dictionaries
        
        feature_dict[activity]
            will give a dictionary of data of all users for that activity
        
        feature_dict[activity][name]
            will give an observation sequence, with shape (n_windows, n_features)
            This is the output of the extract_features function

    Returns
    -------
    models: a dictionary of models, for each activity
    """
    models = {}
    n_components = 3
    
    for activity, activity_data_dict in feature_dict.iteritems():
        Xs = None
        for name, X in activity_data_dict.iteritems():
            Xs = vStackMatrices(Xs, X)
        models[activity] = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
        models[activity].fit(Xs)

    return models