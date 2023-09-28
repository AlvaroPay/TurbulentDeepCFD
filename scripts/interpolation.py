import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import faiss
import cuspatial, cudf

def read_vtu(vtu_path):   
    # Create a reader for the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()

    # Extract the output
    data = reader.GetOutput()

    # Extract point data for UMean and pMean
    u_vtk_array = data.GetPointData().GetArray('UMean')
    p_vtk_array = data.GetPointData().GetArray('pMean')

    # Convert VTK arrays to numpy arrays
    u_np = vtk_to_numpy(u_vtk_array)
    ux_np = u_np[:, 0]  # First component for ux
    uy_np = u_np[:, 1]  # Second component for uy
    p_np = vtk_to_numpy(p_vtk_array)  

    # Extract the points (coordinates) from the VTU data
    points = data.GetPoints()
    x = [points.GetPoint(i)[0] for i in range(points.GetNumberOfPoints())]
    y = [points.GetPoint(i)[1] for i in range(points.GetNumberOfPoints())]

    data = np.column_stack((x, y, ux_np, uy_np, p_np))
    
    if np.any(ux_np > 100):
        raise ValueError

    return data

def resize(data, xmin, xmax, ymin, ymax):
    """ Reduce the size of the computational domain from the initial size of
    (xmin=-10, xmax=30, ymin=-10, ymax=10).

    Parameters
    ----------
    data: numpy.ndarray
          raw average flow field
    xmin: float
          minimum bound upstream of wing geometry
    xmax: float
          maximum bound downstream of wing geometry
    ymin: float
          minimum bound below of wing geometry
    ymax: float
          minimum bound above of wing geometry

    Returns
    -------
    data: numpy.ndarray
          average flow field with new bounds

    """

    xmask = np.logical_and(data[:, 0] >= xmin, data[:, 0] <= xmax)
    data = data[xmask]
    ymask = np.logical_and(data[:, 1] >= ymin, data[:, 1] <= ymax)
    data = data[ymask]
    coord = data[:,0:2]
    fields = data[:,2:]

    return coord, fields

def interpolate(xmin, xmax, ymin, ymax, k, resolution, gpu_id, bluff, data, mach, p: int = 2):
    """
    Interpolate from the base mesh onto a new mesh for the given coordinate
    points and field data values

    Parameters
    ----------
    x_range: int
            xmin and xmax of cropped domain
    y_range: int
            ymin and ymax of cropped domain
    k: int
            parameter of nearest neighbours alogorism
    resolution: int
            mesh size
    bluff: numpy.ndarray
            bluff geometry in the .stl file format
    p: int (default = 2)
        power parameter
    gpu_id: int
        GPU ID

    Returns
    -------
    target_data: numpy.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format [x y pMean UxMean UyMean]
    """

    nx = resolution
    ny = resolution
    xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)].reshape(2, -1).T

    # find nearest neighbours and indices
    xb, yb = resize(data, xmin, xmax, ymin, ymax)
    dist, idx = find_knn(xb, xq, k, gpu_id)

    # calculate inverse distance weighting
    weights = np.power(np.reciprocal(dist, out=np.zeros_like(dist),where=dist != 0), p)

    # set divisor where number is zero to very small number to avoid error
    divisor = np.where(np.sum(weights, axis=1) == 0, 1e-23, np.sum(weights, axis=1))
    yq = np.einsum('ij,ijk->ik', weights, yb[idx]) / divisor.reshape(xq.shape[0], -1)

    target_data = np.concatenate((xq, yq), axis=1)
    bluff_points_idx = find_points_inside(xq, bluff)
    sdf_dist, _ = find_knn(bluff, target_data[:, 0:2], 1, gpu_id)

    sdf_geo_distance = np.copy(sdf_dist)
    sdf_geo_distance[bluff_points_idx] = -1
       
    sdf_geo_2D = sdf_geo_distance.reshape(nx, ny)
    
    flow_reg = generateFlowRegion(sdf_geo_2D)

    # set values for all fields inside wing geometry to zero
    target_data[bluff_points_idx, 2:] = 0
    
    # Define thermodynamic properties of air at ICAO standard atmosphere
    T0 = 288.15  # [K] Total temperature
    p0 = 101325  # [Pa] Total pressure
    gamma = 1.4  # [-] Ratio of specific heats
    R = 287.058  # [J/(kg*K)] Specific gas constant for dry air
    T = T0 / (1 + 0.5 * (gamma - 1) * mach ** 2)
    p_inf = p0 * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (-gamma / (gamma - 1))
    u_inf = mach * np.sqrt(gamma * R * T)

    # Normalise pressure by freestream pressure
    target_data[:, 4] /= (p_inf/1.223)
    
    # Normalise velocities by freestream velocity
    target_data[:, 2] /= u_inf
    target_data[:, 3] /= u_inf
    
    #target_data[bluff_points_idx, :2] = 0
    #test = target_data[:, :2]

    return target_data, sdf_geo_2D, flow_reg


def find_knn(xb, xq, k, gpu_id):
    """
    Find k-nearest neighbours for a query vector based on the input coordinates
    using GPU-accelerated kNN algorithm. More information on
    https://github.com/facebookresearch/faiss/wiki

    Parameters
    ----------
    xb: numpy.ndarray
        coordinate points of raw data
    xq: numpy.ndarray
        query vector with interpolation points
    k: int
       number of nearest neighbours
    gpu_id: int
            ID of GPU which shall be used

    Returns
    -------
    (dist, indexes): (numpy.ndarray, numpy.ndarray)
                     distances of k nearest neighbours, the index for
                     the corresponding points in the xb array

    """

    _, d = xq.shape

    xb = np.ascontiguousarray(xb, dtype='float32')
    xq = np.ascontiguousarray(xq, dtype='float32')

    res = faiss.StandardGpuResources()

    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(xb)
    distances, neighbors = gpu_index.search(xq, k)

    return distances, neighbors


def find_points_inside(target_points, bluff_points):
    """

    Parameters
    ----------
    target_points: numpy.ndarray
                   points of interpolation grid
    wing_points: numpy.ndarray
                 points of wing geometry. Must start and end at the same
                 point in clock- or counterclockwise direction

    Returns
    -------
    points_in_wing_idx: numpy.ndarray
                        indexes of target_points array which are inside wing
    """
    # Convert numpy arrays to GeoSeries for points_in_polygon(args)
    pts = cuspatial.GeoSeries.from_points_xy(
        cudf.Series(target_points.flatten()))
    plygon = cuspatial.GeoSeries.from_polygons_xy(
        cudf.Series(bluff_points.flatten()).astype(float),
        cudf.Series([0, bluff_points.shape[0]]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1])
    )

    # find indexes of points within bluff shape
    df = cuspatial.point_in_polygon(pts, plygon)
    df.rename(columns={df.columns[0]: "inside"}, inplace=True)
    points_in_bluff_idx = df.index[df['inside'] == True].to_numpy()
    return points_in_bluff_idx
    
def generateFlowRegion(sdf_geo_2D, threshold=0):
    """
    Generate a flow region based on the provided sdf_geo_2D.

    Parameters
    ----------
    sdf_geo_2D : numpy.ndarray
        2D signed distance field.
    threshold : float, optional
        Distance threshold to determine inside or outside of the obstacle, by default 0

    Returns
    -------
    numpy.ndarray
        Flow region with values of 0 (inside obstacle) or 1 (outside obstacle).
    """
    
    flow_reg = np.where(sdf_geo_2D <= threshold, 0, 1)
    
    return flow_reg


