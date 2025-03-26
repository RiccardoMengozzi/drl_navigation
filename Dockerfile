FROM osrf/ros:humble-desktop-full



# Create a non-root user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config


# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*


# turtlebot3 dependencies
RUN apt-get update \
    && apt-get install -y \
    ros-humble-gazebo-* \
    # ros-humble-cartographer \
    # ros-humble-cartographer-ros \
    ros-humble-dynamixel-sdk \
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3 \
    && rm -rf /var/lib/apt/lists/*

# stable-baselines3 dependencies
RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install stable-baselines3 
RUN pip install --no-cache-dir --upgrade pip \
    &&  pip install --no-cache-dir \
    --ignore-installed sympy \
    stable-baselines3[extra] 
    # "numpy < 2" \
    # wandb \
    # tabulate 

# Install other used packages
RUN pip install --no-cache-dir --upgrade pip\
    && pip install --no-cache-dir \
    wandb \
    tabulate \
    vim

WORKDIR /ros_ws/src

RUN git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
# pass CACHE_BUST as a build argument to force a cache invalidation
ARG CACHE_BUST
RUN echo "CACHE BUST: ${CACHE_BUST}" && git clone https://github.com/RiccardoMengozzi/drl_navigation.git ${date} 

WORKDIR /ros_ws

USER ros
RUN rosdep update \ 
    && rosdep install -i --from-path src --rosdistro humble -y

USER root

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

#COPY /home/<your_username>/.bashrc /home/${USERNAME}/.bashrc
COPY bashrc /home/${USERNAME}/.bashrc
COPY bashrc /root/.bashrc

