SPHERE = '<?xml version="1.0" ?> \
<sdf version="1.5"> \
  <model name="%s"> \
    <static>true</static> \
    <link name="link"> \
      <pose>0 0 0 0 0 0</pose> \
      <visual name="visual"> \
        <transparency> 0.5 </transparency> \
        <geometry> \
          <sphere> \
            <radius>%s</radius> \
          </sphere> \
        </geometry> \
        <material> \
          <script> \
            <uri>file://media/materials/scripts/gazebo.material</uri> \
            <name>Gazebo/%s</name> \
          </script> \
        </material> \
      </visual> \
    </link> \
  </model> \
</sdf>'