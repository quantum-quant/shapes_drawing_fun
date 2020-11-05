import numpy as np
import pyclipper
from shapely.geometry import LineString

def _cast_to_np_arr_coordinate(input):
    if input is None:
        return None
    elif not isinstance(input, np.ndarray):
        return np.array(input).reshape(len(input), 2)
    else:
        return input.reshape(len(input), 2)

def _shift_to_origin(line):
    """
    find the x shift that shifts a line segment so that
    it goes through the origin
    """
    _check_dim(line)
    if line.shape[0] != 2:
        raise ValueError('a line must be defined by two points')
    x_shift = line[1, 0] - line[1, 1] * (line[1, 0] - line[0, 0]) / (line[1, 1] - line[0, 1])

    return np.array([[x_shift, 0.]])

def _check_dim(coords):
    if len(coords.shape) != 2:
        raise ValueError('Coordinates input array need to have exactly two dimensions')
    if coords.shape[1] != 2:
        raise ValueError('Coordinates should have exactly two columns, x and y')

def translate(coords, shift):
    shift = _cast_to_np_arr_coordinate(shift)
    return coords + shift 

def rotate(coords, radians, origin=None):
    """
    Rotate the input shape defined by coordinates by some angle specified 
    by radians around some origin in the x-y plane. If origin is None, 
    rotate around the center of the shape
    """
    _check_dim(coords)
    origin = _cast_to_np_arr_coordinate(origin)

    cos           = np.cos(radians)
    sin           = np.sin(radians)
    rotate_matrix = np.array([[cos, -sin], [sin, cos]])
    
    if origin is None:
        x_c, y_c = center(coords)
        origin   = np.array([[x_c, y_c]])
        
    coords           = translate(coords, -1 * origin)
    rotated_coords_t = np.dot(rotate_matrix, coords.T)
    rotated_coords   = translate(rotated_coords_t.T, origin)
    return rotated_coords

def reflect(coords, line):
    """
    reflect the input shape defined by coordinates across a line in the x-y plane
    """
    _check_dim(coords)
    line      = _cast_to_np_arr_coordinate(line)
    shift_vec = _shift_to_origin(line)
    vector    = np.diff(line, axis=0) 
    coords    = coords - shift_vec

    vector_norm      = vector / np.dot(vector, vector.T)
    coords_projected = np.dot(coords, vector.T) * vector_norm
    reflected_coords = 2 * coords_projected - coords
    return reflected_coords + shift_vec


def scale(coords, scalar):
    return coords * scalar

def arbitary_transform(coords, transform): 
    """
    apply an arbitary transformation defined by a 2x2 transformation matrix 

    Parameters 
    ----------
    transform:
        a 2x2 array defining the transformation

    Returns
    ----------
    transformed coordinates
    """
    transform = np.array(transform)
    _check_dim(transform)
    _check_dim(coords)

    coords_transformed_t = np.dot(transform, coords.T)
    return coords_transformed_t.T 

def center(coords):
    """
    return the average x, y coordinates as the "center" of a shape
    """
    x     = coords[:, 0]
    y     = coords[:, 1]
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    return x_avg, y_avg

def sort_coordinates(coords, reverse=False):
    """
    sort vertices as specified by the coordinates such that they are in the 
    order of clockwise traversal around the polygon they represent. Counteclockwise 
    if reverse is true
    """
    _check_dim(coords)

    x     = coords[:, 0]
    y     = coords[:, 1]
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    x     = x - x_avg
    y     = y - y_avg

    angles = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi)
    idx    = np.argsort(angles)
    if reverse:
        idx = idx[::-1] 
    return coords[idx]
    

def compute_area(coords):
    """
    compute area of a polygon defined by its coordinates using the Shoelace Algorithm
    https://en.wikipedia.org/wiki/Shoelace_formula
    """
    coords = sort_coordinates(coords)
    coords = np.r_['0, 2', coords, coords[0]]

    pos_ = coords[:-1, 0] * coords[1:, 1] 
    neg_ = coords[1:, 0] * coords[:-1, 1]
    area = 1 / 2. * np.abs(np.sum(pos_ - neg_))
    return area


def compute_perimeter_polygon(coords):
    """
    compute the perimeter of a polygon defined by its coordinates
    """
    coords      = sort_coordinates(coords)
    coords_     = np.r_['0, 2', coords, coords[0]]
    coords_next = coords_[1:] 

    edge_lengths = np.sqrt(np.sum((coords_next - coords) ** 2, axis=1))
    perimeter    = np.sum(edge_lengths) 
    return perimeter

def compute_perimeter_line(coords):
    """
    compute the perimeter of a simple line, i.e., its length
    this is the sum of the line segments between coordinates as is, i.e., NOT sorted.
    """

    deltas       = np.diff(coords, axis=0)
    edge_lengths =  np.sqrt(np.sum(deltas ** 2, axis=1))
    perimeter    = np.sum(edge_lengths)
    return perimeter


def compute_bounds(coords):
    """
    compute the coordinates of the bounding box of a shape
    """
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])

    box_coords = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
    return box_coords 


def compute_offsets(coords, width, **kwargs):
    """
    Use the Shapely library, a python package for manipulation and analysis of geometric objects
    for line dilation
    """
    line    = LineString(coords)
    dilated = line.buffer(width, **kwargs)
    return np.array(list(dilated.exterior.coords))



_BOOL_OPERATION_PYCLIPPER_MAP = {
    'union': pyclipper.CT_UNION,
    'intersection': pyclipper.CT_INTERSECTION
}

def compute_union(coords_a, coords_b):
    return compute_bool_operation(coords_a, coords_b, 'union')

def compute_intersection(coords_a, coords_b):
    return compute_bool_operation(coords_a, coords_b, 'intersection')

def compute_bool_operation(coords_a, coords_b, operation):
    """
    Use the Pyclipper library for optimized polygon boolean operations.
    It is a Cython wrapper for Clipper, an open source library for polygon clipping
    """

    coords_a_scaled = pyclipper.scale_to_clipper(coords_a)
    coords_b_scaled = pyclipper.scale_to_clipper(coords_b)

    pc = pyclipper.Pyclipper()
    pc.AddPath(coords_b_scaled, pyclipper.PT_CLIP, True)
    pc.AddPath(coords_a_scaled, pyclipper.PT_SUBJECT, True)

    solution       = pc.Execute(_BOOL_OPERATION_PYCLIPPER_MAP[operation], pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    coords_clipped = np.squeeze(pyclipper.scale_from_clipper(solution)) 
    return coords_clipped
