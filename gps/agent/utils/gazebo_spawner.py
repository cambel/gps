import os.path
import copy
import rospy
import rospkg

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
)

from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)

from pyquaternion import Quaternion
import numpy as np
from gps.utility.general_utils import get_ee_points

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model("cube")
        delete_model("can")
        # delete_model("cafe_table")
        delete_model("wooden_peg")
        delete_model("peg_board")
        # delete_model("ring")

    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

class GazeboModels:
    """ Class to handle ROS-Gazebo model respawn """

    def __init__(self, models, model_pkg, exit=False):
        if models:
            delete_gazebo_models()

        # Get Models' Path
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        packpath = rospack.get_path(model_pkg)
        self.model_path = packpath + '/models/' 

        self._pub_model_state = rospy.Publisher('/gazebo/set_model_state',
                                        ModelState, queue_size=10)

        self.target_model_pose = None
        if exit:
            rospy.on_shutdown(delete_gazebo_models)

        self._models = models
        self._load_model()


    def _load_model(self, condition=0):
        for m in self._models[condition]:
            if m.file_type == 'urdf':
                self.load_urdf_model(m.name, m.pose, model_reference_frame=m.reference_frame)
            elif m.file_type == 'sdf':
                self.load_sdf_model(m.name, m.pose, model_reference_frame=m.reference_frame)
            elif m.file_type == 'string':
                self.load_sdf_model(m.name, m.pose, string_model=m.string_model, model_reference_frame=m.reference_frame)
        self.cur_models = copy.copy(self._models[condition])
        self.target_model = copy.copy(self._models[condition][0])

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_callback)

    def _gazebo_callback(self, data):
        try:
            ind = data.name.index(self.target_model.name)
            self.target_model_pose = data.pose[ind]
        except ValueError:
            pass

    def get_target_pose(self, target_offset):
        pos = self.target_model_pose.position
        rot = self.target_model_pose.orientation
        pos_arr = np.array([[pos.x, pos.y, pos.z-.93]])
        rot_arr = Quaternion(x=rot.x, y=rot.y, z=rot.z, w=rot.w).rotation_matrix
        return np.ndarray.flatten(get_ee_points(target_offset, pos_arr, rot_arr).T)

    def reset_model(self, condition):
        # TODO: generalize for multiple models
        # For now just reset main model pose
        if self.target_model.name != self._models[condition][0].name:
            for m in self.cur_models:
                self.delete_model(m.name)
            for m in self._models[condition]:
                self.load_urdf_model(m.name, m.pose)
            self.target_model = copy.copy(self._models[condition][0])
            self.cur_models = copy.copy(self._models[condition])
        else:
            for m in self._models[condition]:
                model_state = ModelState(model_name=m.name, pose=m.pose, reference_frame="world")
                self._pub_model_state.publish(model_state)
    
    def update_model_state(self, model, reference_frame="world"):
        model_state = ModelState(model_name=model.name, pose=model.pose, reference_frame=reference_frame)
        self._pub_model_state.publish(model_state)

    def load_urdf_model(self, ref_name, model_pose, model_reference_frame="world"):
        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            spawn_urdf(ref_name, self.load_xml(ref_name, filetype="urdf"), "/",
                                model_pose, model_reference_frame)
        except IOError:
            self.load_sdf_model(ref_name, model_pose, model_reference_frame=model_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def load_sdf_model(self, ref_name, model_pose, string_model=None, model_reference_frame="world"):
        # Spawn model SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            if string_model is None:
                spawn_sdf(ref_name, self.load_xml(ref_name), "/",
                                    model_pose, model_reference_frame)
            else:
                spawn_sdf(ref_name, string_model, "/",
                                    model_pose, model_reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

    def load_xml(self, model_name, filetype="sdf"):       
        # Load File
        with open (self.model_path + model_name +"/model.%s"%filetype, "r") as table_file:
            return table_file.read().replace('\n', '')

    def delete_model(self, model_name):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model(model_name)
        except rospy.ServiceException, e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))
