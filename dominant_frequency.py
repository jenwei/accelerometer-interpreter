from __future__ import division
from autocorr import autocorr
from numpy import argmax

def find_dominant(seg):
    """Finds the dominant frequency of a given ThinkDSP Wave object
    by Meg
    
    Arguments
    ---------
    seg: ThinkDSP Wave Obj
	a segment of Wave
    
    Returns
    -------
    dominant_freq: a Number
	the dominant frequency, relative to the framerate parameter specified in        the Wave.
    """
    lags, corrs = autocorr(seg)
    #argmax() gedom_freq_index = numpy.argmax(corrs[1:len(corrs)])+1
    dom_freq_index = argmax(corrs[1:len(corrs)])+1
    period = dom_freq_index/seg.framerate
    dom_freq = 1/period
    return dom_freq
 
