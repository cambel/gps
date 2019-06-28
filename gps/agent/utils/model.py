import copy
from geometry_msgs.msg import (
    Pose,
    Point,
    Quaternion
)

class Model(object):
    """ Model object """
    def __init__(self, name, position, orientation=[0,0,0,1], file_type='urdf', string_model=None, reference_frame="world"):
        """
        Model representation for Gazebo spawner
        name string: name of the model as it is called in the sdf/urdf
        position array[3]: x, y, z position
        orienation array[4]: ax, ay, az, w
        file_type string: type of model sdf, urdf, or string
        string_model string: full xml representing a sdf model
        reference_frame string: frame of reference for the position/orientation of the model 
        """
        self.name = name
        self.posearr = position
        self.orietarr = orientation
        self.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]) if isinstance(orientation, list) else Quaternion()
        self.pose = Pose(position=Point(x=position[0], y=position[1], z=position[2]), orientation=self.orientation)
        self.file_type = file_type
        self.string_model = string_model
        self.reference_frame = reference_frame

    def get_rotation(self):
        return self.orietarr

    def get_pose(self):
        return self.posearr
