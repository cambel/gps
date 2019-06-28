GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.

CBeltran
Useful setups for the .bashrc file

``` 
source ~/ros_ws/devel/setup.bash

export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh

alias gps='python ~/gps/gps/gps_main.py'
alias gps_test='python ~/gps/gps/test_hyperparameters.py'
alias run_pol='python ~/gps/gps/run_policy.py'

alias ur3='roslaunch ur3_gazebo ur3e.launch'

source /usr/share/gazebo/setup.sh
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ws_ur5/src/ur3/ur3_gazebo/models/

export PYTHONPATH=~/gps:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```
