import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch

parser = argparse.ArgumentParser(description='Generate Surface')
parser.add_argument('--file', type=str, default = "",
                   help='filename', required = True)

def rbfReconstruction(input_point_cloud_filename, epsilon = 1e-4):
    """
    surface reconstruction with an implicit function f(x,y,z) computed
    through RBF interpolation of the input surface points and normals
    input: filename of a point cloud, parameter epsilon
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

    # toy implicit function of a sphere - replace this code with the correct
    # implicit function based on your input point cloud!!!
    # IF = (X - (max_dimensions[0] + min_dimensions[0]) / 2) ** 2 + \
    #      (Y - (max_dimensions[1] + min_dimensions[1]) / 2) ** 2 + \
    #      (Z - (max_dimensions[2] + min_dimensions[2]) / 2) ** 2 - \
    #      (max(bounding_box_dimensions) / 4) ** 2

    ''' ============================================
    #            YOUR CODE GOES HERE
    ============================================ '''
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    IF = np.zeros(shape=(Q.shape[0], 1)) #this is your implicit function - fill it with correct values!
    fp = [0]*3*R.shape[0] #3N weights
    fp[R.shape[0]:R.shape[0]*2] = [epsilon]*R.shape[0]   #for p + eps*n
    fp[R.shape[0]*2:] = [-1*epsilon]*R.shape[0]   #for p - eps*n
    fp = np.array(fp)
    weights = np.zeros(3*R.shape[0])
    offsets_1 = [ p + epsilon * normals[i] for i, p in enumerate(R)]
    offsets_2 = [ p - epsilon * normals[i] for i, p in enumerate(R)]

    all_points = np.concatenate((R, offsets_1, offsets_2))   #3N points
    spline_phi = np.zeros((all_points.shape[0], all_points.shape[0]))
    for i in range(all_points.shape[0]):
        pi = all_points[i]
        for k in range(all_points.shape[0]):
            ck = all_points[k]
            r = np.linalg.norm(pi - ck) + 1e-8
            spline_phi[i][k] = (r**2) * np.log(r)

    weights = np.linalg.solve(spline_phi, fp)

    IF = np.zeros(shape=(Q.shape[0], 1))

    for i in range(Q.shape[0]):
        pi = Q[i]
        pi = pi.reshape((1,3))
        pi = np.repeat(pi, all_points.shape[0], axis=0)
        r_vector = np.linalg.norm(pi - all_points, axis=1) + 1e-8
        spline_phi_pi = (r_vector**2) * np.log(r_vector)
        IF[i] = np.dot(spline_phi_pi, weights)

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
    rbfReconstruction(args.file)
