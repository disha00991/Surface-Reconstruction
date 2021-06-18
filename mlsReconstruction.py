import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch

parser = argparse.ArgumentParser(description='Generate Surface')
parser.add_argument('--file', type=str, default = "",
                   help='filename', required = True)

def mlsReconstruction(input_point_cloud_filename):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points 
    input: filename of a point cloud
    output: reconstructed mesh
    """

    #load the point cloud
    data = np.loadtxt(input_point_cloud_filename)
    points = data[:,:3]
    normals = data[:,3:]


    # construct a 3D NxNxN grid containing the point cloud
    # each grid point stores the implicit function value
    # set N=16 for quick debugging, use *N=64* for reporting results
    N = 64
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points
    bounding_box_dimensions = max_dimensions - min_dimensions # compute the bounding box dimensions of the point cloud
    grid_spacing = max(bounding_box_dimensions)/(N-9) # each cell in the grid will have the same size
    X, Y, Z =np.meshgrid(list(np.arange(min_dimensions[0]-grid_spacing*4, max_dimensions[0]+grid_spacing*4, grid_spacing)),
                         list(np.arange(min_dimensions[1] - grid_spacing * 4, max_dimensions[1] + grid_spacing * 4,
                                    grid_spacing)),
                         list(np.arange(min_dimensions[2] - grid_spacing * 4, max_dimensions[2] + grid_spacing * 4,
                                    grid_spacing)))
    
    # IF = np.zeros(shape=X.shape)
    # toy implicit function of a sphere - replace this code with the correct
    # implicit function based on your input point cloud!!!
    # IF = (X - (max_dimensions[0] + min_dimensions[0]) / 2) ** 2 + \
    #      (Y - (max_dimensions[1] + min_dimensions[1]) / 2) ** 2 + \
    #      (Z - (max_dimensions[2] + min_dimensions[2]) / 2) ** 2 - \
    #      (max(bounding_box_dimensions) / 4) ** 2
    
    # idx stores the index to the nearest surface point for each grid point.
    # we use provided knnsearch function
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    K = 20
    idx = knnsearch(Q, R, K)
    IF = np.zeros(shape=(Q.shape[0],1))

    ''' ============================================
    #            YOUR CODE GOES HERE
    ============================================ '''

    nearest_surface_point_to_self = knnsearch(R, R, 1)
    linalg = np.linalg.norm(R - R[nearest_surface_point_to_self][:,0], axis=1)
    distances_to_nearest_points = np.linalg.norm(R - R[nearest_surface_point_to_self][:,0], axis=1)
    beta = 2 * np.mean(distances_to_nearest_points)
    print(beta)

    for j in range(Q.shape[0]):
        sum_phi = 0
        for i in range(20):
            normal = normals[idx[j]][i]
            p_minus_pi = Q[j] - R[idx[j]][i]
            di_p = normal[0] * p_minus_pi[0] + normal[1] * p_minus_pi[1] + normal[2] * p_minus_pi[2]
            phi = np.exp(-1*(np.linalg.norm(p_minus_pi)**2)/beta**2)
            IF[j] += di_p*phi
            sum_phi += phi
        IF[j] = IF[j]/(sum_phi)

    IF = IF.reshape(X.shape)
    ''' ============================================
    #              END OF YOUR CODE
    ============================================ '''

    verts, simplices = measure.marching_cubes_classic(IF, 0)
    
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title="Isosurface")
    plotly.offline.plot(fig)

if __name__ == '__main__':
    args = parser.parse_args()
    mlsReconstruction(args.file)

