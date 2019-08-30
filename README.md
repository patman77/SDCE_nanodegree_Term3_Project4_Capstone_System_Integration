## Final Capstone Project SDC System Integration
### Team: Neural Riders On The Storm
### Team Members:
* Team Lead: Patrick Klie <info@patman-industries.com>
* Team Member 1: Subramanian Elavathur Ranganath "Subbu" <arjunnaru@gmail.com>
* Team Member 2: Aleksandr Fomenko <xanderfomenko@gmail.com>
* Team Member 3: Jian Li <lijian2005lj@163.com>
* Team Member 4: Yi Yang <davidyy@umich.edu>
* (Consulting Team Member 5: Ahmed Madbouly <ahmedmadbouly@yahoo.com>)


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Introduction

Approaching the final capstone project, we formed our above mentioned team called "Neural Riders On The Storm" consisting of 4 regular team members (Team Lead + Team Member 1-4) with an additional "consultant", who already passed the SDC Nanodegree before. We have the offifical statement from the Udacity Support that also 6 members instead of 5 are allowed. If this should be an issue, please omit the last consulting member for this team submission as he already has the SDC Nanodegree certificate.

### Installation

After having analyzed the possibilities Docker, web-based Workspace, Udacity Virtual Machine and native installation, we finally decided for a mixture of the two latter ones: We converted the Udacity VM to a physical device and booted from that device (we just used an external SSD).

By this, we could overcome many disadvantages of the other possibilities:

* Docker: many of our team members were not really familiar with that method and have seldomly used it throughout the course
* Web-based Workspace: Limited GPU time, needs internet access
* Udacity Virtual Machine: pretty slow with VirtualBox and also with Parallels, no GPU support in the guest system
* Native installation: Necessary to install ROS manually

We described this approach also on "Knowledge" under the link https://knowledge.udacity.com/questions/53477.<br>
Essentially, it's these steps:

1. Get the Udacity VM: wget -c https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Udacity_VM_Base_V1.0.0.zip

2. Depack: unzip Udacity_VM_Base_V1.0.0.zip

3. Untar: tar xvf Ubuntu\ 64-bit\ Udacity\ VM\ V1.0.0.ova

4. Convert to image: VBoxManage clonehd Ubuntu_64-bit_Udacity_VM_V1.0.0-disk1.vmdk disk.img --format RAW (takes some time)

5. Attach an empty external hard drive and remember device, e.g. /dev/sdc (CAUTION: no typos! Otherwise system can get broken)

6. sudo dd if=./disk.img of=/dev/sdX (with remembered device from above, e.g. /dev/sdc, also takes some time)

7. Optionally enlarge partition: sudo gparted (good for large rosbag files)

8. In BIOS: change boot order so that external drive is preferred

9. Boot from this external drive

10. (Optionally): Install NVIDIA GPU driver, simulator and DL based tl detection/classification will run faster

This method was also motivated by the fact that we had a lot of issues with latencies, high CPU load etc. in the beginning when we used VM only.

### Concepts

We all went through the classroom lessons and implemented all the stuff presented in the video walkthroughs including the ROS nodes waypoint updater, DBW. One essential part remaining was the traffic light and detection. For the simulation we recognized that traffic lights are available as 'ground truth', which worked fine, compare the following video:

[Traffic Light Detection in the simulator](./results/capstone-submission01-2019-08-29_05.25.15_h264.mp4)

For real camera images, we though about two possibilities:
* using a classical approach, namely HOG+SVM, as we already implemented in Term 1 for the vehicle detection
* using one of the deep learning approaches
  * specialized model trained with a dedicated set of images with traffic lights
  * pre-trained model

We chose deep learning, as from literature it is well known that very good results can be achieved. Also we wanted to avoid parameter tuning, which is often the case for classical methods.

Although we had some GPU available for training, also on a GPU cluster, we decided to use pretrained models. From the object detection lab in the course, we used the existing approaches from the "Tensorflow  detection model zoo".

### Implementation

##
### Optimization and System Design

From the detection lab in the course, we initially tried the DL models 


### Lessons Learned




The rest of this document we kept to have it as a reference and a tutorial for playing the rosbag files.

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
