import numpy as np

def create_rectangle(src, dst, diameter):
    """
    Create rectangular polygon along src-dst line and with
    width of diameter

    Parameters
    ----------
    src: array_like, shape (2, )
        The coordinates of the start point of the line.
    dst: array_like, shape (2, )
        The coordinates of the end point of the line. The destination point
        is included in the profile, in contrast to standard numpy indexing.
    diameter: int
        rectangle diameter (small length)

    Returns
    -------
    rectangle: 2d array
        coordinates of rectangle. First and last coordinates are identical
    
    """
    vec1 = np.array(dst)-np.array(src)
    vec1 = vec1 / np.linalg.norm(vec1)
    
    x=np.array([0.6,0.5])
    x -= x.dot(vec1) * vec1    # make it orthogonal to k
    x /= np.linalg.norm(x)  # normalize it
    
    rectangle = np.stack([
        np.array(src),
        np.array(src)+x*diameter/2,
        np.array(dst)+x*diameter/2,
        np.array(dst)-x*diameter/2,
        np.array(src)-x*diameter/2,
        np.array(src)])
    
    return rectangle