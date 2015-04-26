def magnitude(xyz):
    """Calculates the magnitude of the accleration vectors
    
    Arguments
    ---------
    xyz: shape (n_samples, 3)
        matrix with the row as the sample, and the columns as x, y, z
        
    Returns
    -------
    magnitude: shape (n_samples,)
        array of the magnitude of each xyz accelerometer sample
    """
    return np.linalg.norm(xyz, axis=1)
