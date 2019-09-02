## Final Capstone Project SDC System Integration

![](results/start.gif)

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

### Organisation

We created a slack channel "neuralriders_ot_storm" to communicate. Slack also offers desktop and mobile apps so we could communicate very efficiently.
Additionally, we created a trello board (http://www.trello.com) to organize our tasks, including due dates, and a timeline.

### Installation

After having analyzed the possibilities Docker, web-based Workspace, Udacity Virtual Machine and native installation, we finally decided for a mixture of the two latter ones: We converted the Udacity VM to a physical device and booted from that device (e.g. from an external SSD).

By this, we could overcome many disadvantages of the other possibilities:

* Docker: many of our team members were not really familiar with that method and have seldomly used it throughout the course
* Web-based Workspace: Limited GPU time, needs internet access
* Udacity Virtual Machine: pretty slow with VirtualBox and also with Parallels, no GPU support in the guest system
* Native installation: Necessary to install ROS manually, no support from Udacity

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

This method was also motivated by the fact that we had a lot of issues with latencies, high CPU load etc. in the beginning when we used VM only. This led to instabilities and latencies in the car's steeering, resulting in leaving the lane or even going off-road.

### Concepts

We all went through the classroom lessons and implemented the concepts presented in the video walkthroughs including the ROS nodes waypoint updater, DBW, and later the traffic light detection. One essential part remaining was exactly this traffic light detection and classification. For the simulation we recognized that traffic lights are available as 'ground truth', which worked fine, compare the following video:

[![Traffic Light Detection in the simulator](./results/capstone-submission01-2019-08-29_05.25.15_h264_00000071.png)](https://youtu.be/PU90_rMJark)

For real camera images without the available ground truth information from the simulator, we thought about two possibilities:
* using a classical approach, namely HOG+SVM, as we already implemented in Term 1 for the vehicle detection
* using one of the deep learning approaches
  * specialized model trained with a dedicated set of training images with traffic lights
  * pre-trained model

We chose deep learning, as from literature it is well known that very good results can be achieved with DL. Also we wanted to avoid parameter tuning, which is often the case for classical methods.

Although we had some GPUs available for training (also on a GPU cluster), we decided to use pretrained models, also due to the lack of remaining time. From the object detection lab in the course, we used the existing approaches from the "Tensorflow detection model zoo".

### Implementation

All the concepts have been implemented into the ROS nodes, as proposed in the classroom. For better understanding, we visualized the overall node / message layout with rqt_graph contained in ROS kinetic:

![Traffic Light Detection in the simulator](./Doc/ROS_System_Overview/rqt_graph_arjunnaru.jpg)

##
### Optimization and System Design

From the detection lab in the course, we initially tried the DL models to get a first understanding how traffic light detection. We derived the runtimes on a laptop GPU just to get a rough understanding of performances and runtimes. We gained these initial insights:
* "ssd_mobilenet_v1" from 2017, < 75 ms/frame
* "rfcn_resnet101", <110 ms/frame
* "faster_rcnn_inception_resnet_v2", <450 ms/frame

Later we switched to desktop GPUs to be comparable to the Titan X in Carla.

From the following video it can be seen that the better the model, the better the detections are, coming with higher runtimes (see above).

[![tl detection on real world examples](./results/all_00000027.png)](https://youtu.be/xgD799cP8xs)

Nevertheless we used detector pre-trained on COCO dataset from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) — [ssd_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) for lower runtimes in the beginning for 2 reasons:
* we started with tensorflow without GPU support, giving us over 400% CPU load
* also on potential reviewers computers, there could be no GPU support, and we wanted to avoid that the reviewers system gets overloaded while testing on the simulator

Additionally, we only analyze every Nth frame (N=2) to be on the safe side.

As you can notice, our model was trained with Tensorflow 1.12. Thus, we converted it to be compatible with Tensorflow 1.3 using this  [solution](https://stackoverflow.com/questions/53927700/how-to-use-object-detection-api-with-an-old-version-of-tensorflow-v1-3-0). The same was done to other models, mentioned in this README.

## Results

### Simulator
Finally, the entire system works as expected, which can be seen in the following video:

[![Final run in the simulator with DL based tl detection](./results/full_run_gpu_h264_00000170.png)](https://youtu.be/pw6uTMEGYgM)

An additional ROS topic "/image_color_detect", which can be seen in the top left corner shows the current detections, together with the confidences. The rqt topic monitor in the middle clearly shows that all messages related to vehicle steering are approximately at the desired 50Hz. On the right side we see the GPU load of 40%, rendered with "nvidia-smi". On that particular system, we had tensorflow-gpu on an NVIDIA GeForce GTX 1060, so below the hardware equipment of Carla with a Titan X. On the lower right "top" shows the approx. CPU loads:

1. The simulator with 130%
2. styx server 65%
3. traffic light detection 60%

### Real world examples

We also tested the traffic light detection on real word examples, given by the supplied rosbag files:

[![tl detection on 1st rosbag file](./results/ssd-rosbag_justtrafficlight-2019-09-02_01.49.51_00000670.png)](https://youtu.be/JrgypaOlld0)

[![tl detection on 2nd rosbag file](./results/ssd-rosbag_loop-2019-09-02_01.55.15_00001120.png)](https://youtu.be/X8ZdUgSqAtk)

[![tl detection on 3rd rosbag file](./results/ssd-rosbag_trainingseq-2019-09-02_02.03.35_00002429.png)](https://youtu.be/ZgElg7v4ePo)

The sequences (especially the last one) are pretty tough because of:
* front window reflections
* dirt on the window
* the traffic light color is not always clearly visible
* exposure time pretty high sometimes
* camera images in the last sequence not sharp

With "ssd_mobilenet_v1", the traffic light detections were correct in case of occurrence, but the availability was rather low.

Therefore, we decided to test further DL object detection models from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to improve the detection rate, insights:

| Name | Inference on Titan X, ms | COCO mAP, % | Comments |
|------|--------------------------|-------------|----------|
| faster_rcnn_resnet101_coco_2018_01_28 | 106 | 32 | detects really good, but only at 5Hz |
| rfcn_resnet101_coco_2018_01_28 | 92 | 30 | good detector, but only about 6Hz |
| faster_rcnn_resnet50_coco_2018_01_28 | 89 | 30 | almost the same as previous |
| faster_rcnn_inception_v2_coco_2018_01_28 | 58 | 28 | similar detection quality, but 2x faster than prev one — 14Hz on a GeForce GTX 1060 |

In the following videos, the better detection results via the faster_rcnn model can be seen:

[![improved tl detection on loop rosbag file](./results/tfgpu-tldetection-lot-2019-09-01_03.34.05_00000040.png)](https://youtu.be/Zgs7yY_50fU)
<br><br>
[![improved tl detection on trafficlight rosbag file](./results/tfgpu-tldetection-lot-2019-09-01_03.34.05_00000040.png)](https://youtu.be/W2gGK-pVUQM)


In conclusion, we perform the tl detection with a simpler ssd_mobilenet_v2 model when running in simulator (roslaunch with styx.launch), and switch to a "faster_rcnn_inception_v2 model" when running on Carla (roslaunch with site.launch).

### Lessons Learned

We learned that the entire system heavily depends on the available hardware. Especially, if the GPU is enabled, tensorflow can benefit from it, and the overall runtime improves heavily.

<br>
The rest of this document is the original README.md that we kept as a reference and a tutorial for playing the rosbag files. Except 1 thing, for testing with rosbag files, we created special launch file, so you need to launch bag.launch, instead of site.launch.

<br><br><br>















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

### Rosbag testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in bag mode
```bash
cd CarND-Capstone/ros
catkin_make
source devel/setup.sh
roslaunch launch/bag.launch
```
5. Confirm that traffic light detection works on real life images by looking on topic /image_color_detect with rqt ImageView, or rviz.

### Carla running (Udacity Self-Driving Car)
1. Launch Carla
2. Launch project in site mode
```bash
cd CarND-Capstone/ros
catkin_make
source devel/setup.sh
roslaunch launch/site.launch
```
But you know better :)

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
