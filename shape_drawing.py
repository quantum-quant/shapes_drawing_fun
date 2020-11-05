import collections
import functools
from abc import ABC, abstractmethod
from functools import partialmethod

import numpy as np
import matplotlib.pyplot as plt

from geom_computation import (translate, rotate, reflect, scale, compute_offsets, compute_intersection, 
                              compute_union, compute_area, compute_bounds, compute_perimeter_line,
                              compute_perimeter_polygon, arbitary_transform)


def _curry_func(func):
    """
    Instance method decorator for currying transformation functions
    """
    @functools.wraps(func)
    def wrapper(shape, *args, **kwargs):
        parent_node             = func(shape, *args, **kwargs)
        _class                  = shape.__class__
        if parent_node.child_cls is not None:
            _class = parent_node.child_cls
        new_shape               = shape.curry(_class, parent_node.compute(), parent_node.new_attrs)
        shape.child             = new_shape
        new_shape.parent        = parent_node  
        new_shape.parent.output = new_shape 
        new_shape.gen_num       = shape.gen_num + 1
        return new_shape
    return wrapper

#------------------------------------------------------------------------------------------------------

TransformTreeDesc = collections.namedtuple('TransformTreeDesc', ['gen_num', 'inputs', 'transform'])

class TransformNames:
    """
    A bookkeeping class for all possible transformations and their string constants
    """
    translate = 'TRANSLATE'
    rotate    = 'ROTATE'
    reflect   = 'REFLECT'
    union     = 'UNION'
    intersect = 'INTERSECT'
    scale     = 'SCALE'
    offset    = 'OFFSET'
    arbitary  = 'ARBITARY'

class TransformNode(ABC):
    """
    Abstract class for a transform node. A transform node 
    can be thought of as the parent of a transformed shape.

    Attributes
    ---------
    type: string
        type of transform, e.g., translation or rotation
    inputs: sequence
        a sequence of inputs that go into the transformation, an input
        can be a shape or a constant vector or matrix
    output: Polygon, Line, DilatedLine,
        an instance of a derived class of _BaseShape
    child_cls: class 
        specifies the child class if different from input shape's class
    new_attrs: dict
        a dictionary of additional attributes that the transformed
        shape may be initialized with
    """


    def __init__(self, transform_type, *inputs):
        """
        Parameters
        ----------
        transform_type: string
            type of transform, e.g., translation or rotation etc

        inputs: tuple
            a list of inputs that go into the transformation, an input
            can be a shape or a constant vector or matrix
        """

        self.type        = transform_type
        self.inputs      = list(inputs)
        self.output      = None
        self.child_cls   = None
        self.new_attrs   = {}

    @abstractmethod
    def compute(self):
        pass

    def __repr__(self):
        return f'{self.type}-->{self.output}'


class Translate(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.translate, *inputs)

    def compute(self):
        coords     = self.inputs[0].coordinates_np
        shift      = self.inputs[1]
        new_coords = translate(coords, shift)
        return new_coords
        
class Rotate(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.rotate, *inputs)

    def compute(self):
        coords     = self.inputs[0].coordinates_np
        angle      = self.inputs[1]
        origin     = self.inputs[2]
        new_coords = rotate(coords, angle, origin=origin)
        return new_coords

class Reflect(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.reflect, *inputs)

    def compute(self):
        coords     = self.inputs[0].coordinates_np
        vector     = self.inputs[1]
        new_coords = reflect(coords, vector)
        return new_coords

class Scale(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.scale, *inputs)

    def compute(self):
        coords     = self.inputs[0].coordinates_np
        scale_fac  = self.inputs[1]
        new_coords = scale(coords, scale_fac)
        return new_coords

class Offset(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.offset, *inputs)
        self.child_cls = DilatedLine

    def compute(self):
        coords                  = self.inputs[0].coordinates_np
        offset_width            = self.inputs[1]
        self.new_attrs['width'] = offset_width
        new_coords              = compute_offsets(coords, offset_width)
        return new_coords


class Intersect(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.intersect, *inputs)

    def compute(self):
        coords_a   = self.inputs[0].coordinates_np
        coords_b   = self.inputs[1].coordinates_np
        new_coords = compute_intersection(coords_a, coords_b)
        return new_coords


class Union(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.union, *inputs)

    def compute(self):
        coords_a   = self.inputs[0].coordinates_np
        coords_b   = self.inputs[1].coordinates_np
        new_coords = compute_union(coords_a, coords_b)
        return new_coords

class Arbitary(TransformNode):

    def __init__(self, *inputs):
        super().__init__(TransformNames.arbitary, *inputs)

    def compute(self):
        coords    = self.inputs[0].coordinates_np
        transform = self.inputs[1]
        new_coords = arbitary_transform(coords, transform)
        return new_coords


def _post_order_traversal(shape, results_out):
    inputs = []
    if shape.parent is not None:
        for _input in shape.parent.inputs:
            if isinstance(_input, _BaseShape):
                _post_order_traversal(_input, results_out)
            inputs.append(_input)
        node_description = TransformTreeDesc(shape.gen_num, inputs, shape.parent) 
        results_out.append(node_description)

#------------------------------------------------------------------------------------------------------
    
_global_shape_type_register = []

_DEFAULT_COLOR = 'blue'
_DEFAULT_LAYER = 10

class _BaseShape:
    """
    Base class for all shapes

    Attributes
    ----------
    _coordinates: numpy array
        a numpy array of [x, y] coordinate pairs
    name: string
        name of the shape object
    color: any matplotlib acceptable color constructs
        color of the shape
    layer: int
        layer of the shape
    parent: TransformNode
        points to the parent transform that generated the object
    child: _BaseShape derived object 
        child shape that the object is directly responsible for
    gen_num: int
        generation number; tells which generation the object is on the transform tree
    """

    _relational_attrs = ['parent', 'child', 'gen_num']
    _basic_attrs      = ['_coordinates', 'name', 'color', 'layer']

    def __init_subclass__(cls):
        _global_shape_type_register.append(cls)

    def __init__(self, name, coordinates, color=_DEFAULT_COLOR, layer=_DEFAULT_LAYER):
        self._coordinates = np.array(coordinates)
        self.name         = name
        self.color        = color
        self.layer        = layer
        self.parent       = None
        self.child        = None
        self.gen_num      = 0

    @property
    def coordinates(self):
        return self._coordinates.tolist()
    
    @property
    def coordinates_np(self):
        return self._coordinates

    @property
    def top(self):
        """
        Reference to the top/root of the transform tree, i.e., the shape after the final transformation 
        Canvas only has access to this
        """
        _top = self
        child = self.child
        while child is not None:
            _top = child
            child = child.child
        return _top

    def view(self, viewer, **plot_kw):
        """
        Plot/visualize the shape

        Parameters
        ---------
        viewer: a derived ShapeViewer object 
            viewer is responsible for actually rendering the shape 
            and displaying to the user
        """
        viewer.show(self, **plot_kw)

    def traverse_transform_tree(self):
        """
        Perform a post-order traversal, i.e. from parents to children, 
        of the transformations that lead to the object

        Returns
        -------
        A list of TransformTreeDesc namedtuples that fully describe
        each level of the tree
        """

        transform_tree = []
        _post_order_traversal(self, transform_tree)
        return transform_tree

    def __repr__(self):
        return f'shape object <name:{self.name}|type:{self.__class__.__name__}|layer:{self.layer}|gen_num:{self.gen_num}>'
    
    def curry(self, _class, new_coords, attributes):
        if isinstance(_class, _BaseShape):
            raise RuntimeError('can only curry functions of derived classes of _BaseShape')
        new_shape = _class(self.name, new_coords, color=self.color, layer=self.layer)
        all_attrs = {**self.__dict__, **attributes} 

        for _attr, val in all_attrs.items():
            if _attr not in _BaseShape._relational_attrs and _attr not in _BaseShape._basic_attrs:
                setattr(new_shape, _attr, val)
        return new_shape
    
    @_curry_func
    def translate(self, shift):
        """
        Translate the shape by a vector specified by shift

        Parameters
        ----------
        shift: sequence
            one [x,y] coordinate pair that specifies the shift

        Returns
        -------
        A Translate TransformNode that is then curried by the @curry_func decorator  
        """

        return Translate(self, shift) 

    @_curry_func
    def rotate(self, radians, origin=None):
        """
        Rotate the shape 

        Parameters
        ----------
        radians: float
            rotation angle in radians

        origin: sequence
            one [x, y] coordinate pair that specifies the center of rotation.
            if origin is None, rotate around (0, 0)

        Returns
        -------
        A Rotate TransformNode that is then curried by the @curry_func decorator  
        """
        return Rotate(self, radians, origin)

    @_curry_func
    def reflect(self, vector):
        """
        Reflect the shape across a line in the x-y plane

        Parameters
        ----------
        vector: sequence
            a pair of [x, y] coordinate pairs that specifies the line
            of reflection

        Returns
        -------
        A Reflect TransformNode that is then curried by the @curry_func decorator  
        """
        return Reflect(self, vector)

    @_curry_func
    def scale(self, scale_factor):
        """
        Scale the shape by a scale_factor 

        Parameters
        ----------
        scale_factor: float

        Returns
        -------
        A Scale TransformNode that is then curried by the @curry_func decorator  
        """

        return Scale(self, scale_factor)

    @_curry_func
    def intersect(self, other):
        """
        AND operation on the shapes; find the intersection of the shape object
        with another shape object

        Parameters
        ----------
        other: a _BaseShape derived object

        Returns
        -------
        A Intersect TransformNode that is then curried by the @curry_func decorator  
        """

        return Intersect(self, other)

    @_curry_func
    def union(self, other):
        """
        OR operation on the shapes: find the union of the shape object
        with another shape object

        Parameters
        ----------
        other: a _BaseShape derived object

        Returns
        -------
        A Union TransformNode that is then curried by the @curry_func decorator  
        """
        return Union(self, other)

    @_curry_func
    def arbitary_transform(self, transform):
        """
        Apply an arbitary transformation to the shape by a transformation
        matrix

        Parameters
        ----------
        transform: sequence
            a 2x2 array defining the arbitary transformation matrix 

        Returns
        -------
        A Arbitary TransformNode that is then curried by the @curry_func decorator  
        """
        return Arbitary(self, transform)

    def area(self):
        return compute_area(self.coordinates_np)

    def bounds(self):
        return compute_bounds(self.coordinates_np)


class Polygon(_BaseShape):
    """
    Polygon shape, a closed path of coordinates when rendered
    """
    def perimeter(self):
        return compute_perimeter_polygon(self.coordinates_np)
    
class Line(_BaseShape):
    """
    Line shape, a open path of coordinates when rendered

    Attributes
    ---------
    width: float
        always 0 for a line shape
    """
    
    def __init__(self, name, coordinates, color=_DEFAULT_COLOR, layer=_DEFAULT_LAYER):
        super().__init__(name, coordinates, color, layer)
        self.width = 0

    def area(self):
        return 0
        
    def perimeter(self):
        return compute_perimeter_line(self.coordinates_np)
    
    @_curry_func
    def offset(self, width):
        """
        Draw parallel lines some distance away from the line shape so that 
        it has a thickness. Also known as offsetting or dilation in the
        computational geometry literature. 

        Parameters
        ----------
        width: float 
            distance away from the line shape to draw the parallel offset

        Returns
        -------
        A Offset TransformNode that is then curried by the @curry_func decorator  
        """
        return Offset(self, width)

class DilatedLine(Polygon):
    """
    Dilated line, a line with offsetted/dilated borders. This 
    is the resulting shape object when offset is applied to a line shape.
    It can also be called directly to generate a dilated line


    Attribute
    --------
    width: float
        a non zero value that sets the offset
    """
    
    def __init__(self, name, coordinates, width=0.1, color=_DEFAULT_COLOR, layer=_DEFAULT_LAYER):
        super().__init__(name, coordinates, color, layer)
        self.width = width
   
    @_curry_func
    def offset(self, width=None):
        width = self.width if width is None else width 
        return Offset(self, width)


class SmoothCurve(_BaseShape):
    """
    To be extended in the future
    """
    pass


#------------------------------------------------------------------------------------------------------

class Canvas:
    """
    Canvas is the drawing board for shapes, where shapes
    can be drawn on, added, retrieved or deleted. It would 
    be the object to pickle if the user wants to save their
    project

    Attribute
    ---------
    container: OrderedDict
        a ordereddict container that's responsible for storing the shape objects
    """


    def __init__(self):
        self._container = ShapeContainer() 

    def add_shape(self, shape): 
        self._container.store(shape)

    def remove_shape(self, name):
        self._container.remove(name)

    def view(self, viewer, names=None, **plot_kw):
        """
        Plot/visualize shapes in the canvas. Show all
        the shapes when names is None

        Parameters
        ---------
        viewer: a derived ShapeViewer object 
            viewer is responsible for actually rendering the shape 
            and displaying to the user
        """
        names  = names if names is not None else self._container.names
        shapes = self._container.select(names)
        viewer.show(shapes, **plot_kw)

    def get_all_names(self):
        return self._container.names

    def __getitem__(self, name):
        return self._container[name]

    def __repr__(self):
        out = ''
        for _, shape in self._container.items():
            out += f'{shape}\n'     
            out += '---------------------------------------\n'
        out += f'total number of top shape objects in this canvas: {len(self._container)}'
        return out

    def bounds(self, names=None):
        """
        bounding box of the canvas
        taking into account of all the shapes it contains
        """
        names  = names if names is not None else self._container.names
        shapes = self._container.select(names)
        shapes = shapes if isinstance(shapes, list) else [shapes]

        x_min = np.minimum.reduce([shape.bounds()[0,0] for shape in shapes])
        x_max = np.minimum.reduce([shape.bounds()[2,0] for shape in shapes])
        y_min = np.minimum.reduce([shape.bounds()[0,1] for shape in shapes])
        y_max = np.minimum.reduce([shape.bounds()[1,1] for shape in shapes])
        
        canvas_bounds = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
        return canvas_bounds 

def _make_method(cls):
    """
    Produce an instance method for drawing and adding a shape 
    to a canvas object given a shape class. This keeps Canvas 
    class 'automatically' updated with new draw_[new_shape]() whenever we add
    new shape classes
    """
    def _factory_method(self, cls, *args):
        shape = cls(*args)
        self.add_shape(shape)
    f         = partialmethod(_factory_method, cls) 
    f.__doc__ = f'draw {cls.__name__} and add to canvas. See more details in its class'
    return f

for _cls in _global_shape_type_register:
    cls_name = _cls.__name__
    setattr(Canvas, 'draw_' + cls_name.lower(), _make_method(_cls)) 


class ShapeContainer(collections.OrderedDict):
    """
    The container object that is responsible for 
    storing and managing drawn shapes. The container only 
    exposes the "top" node of the shapes 
    """

    def __init__(self):
        super().__init__()
        self.names = self.keys() 
    
    def _check_exist(self, name):
        return name in self.names

    def _assert_new(self, name):
        if self._check_exist(name):
            raise NameError(f'shape with name {name} already exists')
    
    def _assert_exist(self, name):
        if not self._check_exist(name):
            raise NameError(f'shape with name {name} does not exist')

    def __getitem__(self, name):
        shape = super().__getitem__(name)
        return shape.top

    def store(self, shape):
        self._assert_new(shape.name)
        self[shape.name] = shape
        self.names = self.keys()

    def remove(self, name):
        self._assert_exist(name)
        del self[name]
        self.names = self.keys()

    def select(self, names):
        if isinstance(names, str):
            self._assert_exist(names)
            return self[names]
        else:
            _ = [self._assert_exist(n) for n in names]
            return [self[n] for n in names]

#------------------------------------------------------------------------------------------------------

def plot_polygon(shape, axe, **plot_kw):
    color = shape.color
    layer = shape.layer
    name  = shape.name
    x, y  = zip(*shape.coordinates)
    axe.fill(x, y, color=color, zorder=layer, label=name, **plot_kw) 

def plot_line(shape, axe, **plot_kw):
    color = shape.color
    layer = shape.layer
    name  = shape.name
    x, y  = zip(*shape.coordinates)
    axe.plot(x, y, color=color, zorder=layer, label=name, **plot_kw) 


# pylint: disable=arguments-differ

class ShapeViewer(ABC): 
    """
    Abstract class to provide interface for
    any custom implemented viewer object 
    """

    @abstractmethod
    def show(self, *args, **plot_kw):
        pass

class PyplotViewer(ShapeViewer):
    """
    A shape viewer based on pyplot from matplotlib
    """

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize

    @staticmethod
    def plot_shape(shape, axe, **plot_kw):
        if isinstance(shape, DilatedLine): 
            plot_line(shape, axe, **plot_kw)
        elif isinstance(shape, Line):
            plot_line(shape, axe, **plot_kw)
        elif isinstance(shape, Polygon):
            plot_polygon(shape, axe, **plot_kw)
        else:
            raise NotImplementedError
    
    def show(self, shapes, show_label=False, **plot_kw):
        shapes   = shapes if isinstance(shapes, list) else [shapes]
        fig, axe = plt.subplots(figsize=self.figsize)

        for shape in shapes:
            self.plot_shape(shape, axe, **plot_kw)

        axe.set_aspect('equal', 'box') 
        if show_label:
            axe.legend(loc='center left', bbox_to_anchor=(1.05, 1))

        return fig, axe

