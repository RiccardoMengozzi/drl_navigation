# automatic commands at terminal opening
source /opt/ros/humble/setup.bash
. /usr/share/gazebo/setup.sh
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# my aliases
alias cb='colcon build'
alias sc='source install/setup.bash'