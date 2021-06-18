import numpy as np

def knnsearch(Q, R, K):
    """
    KNNSEARCH   Linear k-nearest neighbor (KNN) search
    IDX = knnsearch(Q,R,K) searches the reference data set R (n x d array
    representing n points in a d-dimensional space) to find the k-nearest
    neighbors of each query point represented by eahc row of Q (m x d array).
    The results are stored in the (m x K) index array, IDX. 
    
    Rationality
    Linear KNN search is the simplest appraoch of KNN. The search is based on
    calculation of all distances. Therefore, it is normally believed only
    suitable for small data sets. However, other advanced approaches, such as
    kd-tree and delaunary become inefficient when d is large comparing to the
    number of data points.
    %
    See also, kdtree, nnsearch, delaunary, dsearch

    By Yi Cao at Cranfield University on 25 March 2008
    """

    N, M = Q.shape
    idx = np.zeros((N, K), dtype = int)
    D = np.zeros((N, K))
    fident = np.array_equal(Q, R)
    if K==1:
        for k in range(0, N):
            d = np.sum((R[:, :] - Q[k, :]) ** 2, axis=1)
            if fident:
                d[k] = float('inf')
            D[k] = np.min(d)
            idx[k] = np.argmin(d)
    else:
        for k in range(0, N):
            d = np.sum((R[:, :] - Q[k, :]) ** 2, axis=1)
            if fident:
                d[k] = float('inf')
            D[k, :] = np.sort(d)[:K]
            idx[k, :] = np.argsort(d)[:K]
    print("==>Nearest neighbour search completed!")
    return idx