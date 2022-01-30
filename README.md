# Edge-Cloud Collaborative Real-Time Online Video Object Detection  



## Introduction

**Edge-Cloud Collaborative Real-Time Online Video Object Detection(ECCDetect)**   proposes a fast online video object detection method that leverages the accurate object detector on the Cloud for sparse key frames and handles other video frames by lightweight object tracker on resource limited devices at the system edge in a collaborative manner. It is worth nothing that:

- ECCDetect is the first solution to fast online multi-object detection on real-time video streams based on edge-cloud collaborations. 
- We propose a novel tracking-assisted video object detection solution to reduce the overall processing time.
- We implemented ECCDetect on our real-world ECC prototype and conducted extensive experiments with large-scale video datasets 

This approach is substantially faster than existing detection methods in video with acceptable accuracy loss on the Imagenet VID dataset. 



## Dependencies

1. Python 3.6+
2. Opencv
3. Pytorch 1.0 +
4. torch-vision
5. CUDA 10.0+



## Dataset

Download Imagenet VID 2015 dataset from [here](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php). This is the link for ILSVRC2017 as the link for ILSVRC2015 seems to down now, and get list of training, validation and test dataset in [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing)。



## Installation

First, Install all the python dependencies using pip:

```
$ pip install -r requirements.txt
```

Compile the detection CUDA dependencies using following simple commands(Cloud):

```
$ cd ./ECCDetect_cloud/ecc_cloud/lib/
$ python setup.py build develop
```

Compile the tracking CUDA dependencies using following simple commands(Edge):

```
$ cd ../../../ECCDetect_edge/ecc_edge/
$ python setup.py build_ext --inplace
```



## Train

Make sure to be in python 3.6+ environment with all the dependencies installed. Detector and Tracker are trained separately.

#### Detection

We use resnet101-based RFCN as Detector. 

The nms, roi_pool, roi_align, psroi_pool and psroi_align come from below link. **Thanks for their open source work.**

- NMS, ROIPool, ROIAlign: [jwyangfaster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)
- PSRoIPool, PSRoIAlign: [McDo/PSROIAlign-Multi-Batch-PyTorch](https://github.com/McDo/PSROIAlign-Multi-Batch-PyTorch)

##### 1. Model Preparation

Download pretrained ResNet-101 from [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing)  and put it to `./ECCDetect_cloud/ecc_cloud/pretrained_model/resnet101_rcnn.pth`.

##### 2. Train

```
$ cd ./ECCDetect_cloud/ecc_cloud/
$ CUDA_VISIBLE_DEVICES=$GPU_ID python eccdetect_cloud_train.py \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda 
                   --cag
```

**Note:**

- Set `--s` to identified different experiments. 
- Model are saved to `./models/detect.pth` that you can download from  [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing) 

#### Tracking

We use AlexNet-base SiameseRPN as Tracker. 

The SiamRPN come from below link. **Thanks for their open source work.**

- SiamRPN:  [STVIR/pysot](https://github.com/STVIR/pysot)

##### 1. Model Preparation

Download pretrained backbones from [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing) and put them in `./ECCDetect_edge/ecc_edge/alexnet.pth` directory

##### 2. Train

```
$ cd ./ECCDetect_edge/ecc_edge/
$ CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=2333 \
    --cfg config.yaml\
    ./tools/eccdetect_edge_train.py 
```

**Note:**

- Add Tracker to your PYTHON-PATH  `export PYTHONPATH=/path/to/ecc_edge:$PYTHONPATH`
- Refer to [Pytorch distributed training](https://pytorch.org/docs/stable/distributed.html) for detailed description.
- Model are saved to `./experiments/siamrpn_alex_dwxcorr_multi/track.pth` that you can download from  [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing) 



## Evaluation

After training, you can evaluate ECCDetect on VID val dataset.

Before following steps, you should run **Emqx**  in docker to support the communication between edge and cloud container. The command is: 

```
$ docker run -d --name ecc_broker -p {YOUR_EMQX_PORT}:1883 emqx/emqx:v4.1.1
```

For the edge, we use the following commend to run the `./ECCDetect_edge/ecc_edge/tools/eccdetect_edge_eval.py` and the corresponding docker environment for starting the edge docker

```
$ cd ./ECCDetect_edge/
$ chmod a+rwx ./ecc_edge/eval/
$ docker build -f Dockerfile.eval -t ecc_detect_edge:v1 .
$ . start_edge.sh
```

For the cloud, we use the following commend to run the `./ECCDetect_cloud/ecc_cloud/eccdetect_cloud_eval.py` and the corresponding docker environment for starting the cloud docker and connecting with the edge

```
$ cd ./ECCDetect_cloud/
$ docker build -f Dockerfile.eval -t ecc_detect_cloud:v1 .
$ . start_cloud.sh
```

We use docker finish evaluation while the results will be written in `./ECCDetect_edge/ecc_edge/eval/` and mAP will be computed directly after evaluation.




## Test

You can test ECCDetect on sampled video frames that in `./ECCDetect_edge/ecc_edge/test/images/` .

For the edge, we use the following commend to run the `./ECCDetect_edge/ecc_edge/tools/eccdetect_edge_test.py` and the corresponding docker environment for starting the edge docker

```
$ cd ./ECCDetect_edge/
$ chmod a+rwx ./ecc_edge/test/output/
$ docker build -f Dockerfile.test -t ecc_detect_edge:v1 .
$ . start_edge.sh
```

For the cloud, we use the following commend to run the `./ECCDetect_cloud/ecc_cloud/eccdetect_cloud_test.py` and the corresponding docker environment for starting the cloud docker and connecting with the edge

```
$ cd ./ECCDetect_cloud/
$ docker build -f Dockerfile.test -t ecc_detect_cloud:v1 .
$ . start_cloud.sh
```

After this, the results of test will be written in `./ECCDetect_edge/ecc_edge/test/output/` and it shows the ECCDetect performance on video frames.



## Demo

You can see the demo of ECCDetect on video that in `./ECCDetect_edge/ecc_edge/demo/video.avi` .

For the edge, we use the following commend to run the `./ECCDetect_edge/ecc_edge/tools/eccdetect_edge_demo.py` and the corresponding docker environment for starting the edge docker

```
$ cd ./ECCDetect_edge/
$ chmod a+rwx ./ecc_edge/demo/output/
$ docker build -f Dockerfile.demo -t ecc_detect_edge:v1 .
$ . start_edge.sh
```

For the cloud, we use the following commend to run the `./ECCDetect_cloud/ecc_cloud/eccdetect_cloud_demo.py` and the corresponding docker environment for starting the cloud docker and connecting with the edge

```
$ cd ./ECCDetect_cloud/
$ docker build -f Dockerfile.demo -t ecc_detect_cloud:v1 .
$ . start_cloud.sh
```

After this, the results of demo will be written in `./ECCDetect_edge/ecc_edge/demo/output/` and it shows the ECCDetect performance on video.



## Multiprocessing Test

You can test the performance of multiprocessing implementation in `./ECCDetect_edge/ecc_edge/tools/mutiprocessing_test.py`.

We simulate the multi-object tracking with the template set in advance.

```
$ cd ./ECCDetect_edge/ecc_edge/
$ export PYTHONPATH=/path/to/ecc_edge:$PYTHONPATH
$ python ./tools/mutiprocessing_test.py
```

After this, the results of parallel time and serial time will be shown in the terminal.



## 
## Main Results

|       Type        |          Method           |                           Backbone                           |    Online    |       mAP        |      FPS(cloud)      |      FPS(edge)       |
| :---------------: | :-----------------------: | :----------------------------------------------------------: | :----------: | :--------------: | :------------------: | :------------------: |
|    Flow-Guided    |       FGFA<br />DFF       |              ResNet101+RFCN<br />ResNet101+RFCN              | No<br />Yes  | 63.70<br />72.93 |    7.3<br />13.5     |    0.61<br />3.78    |
|    LSTM-Based     | Bottleneck-LSTM<br />MEGA |       MobileNetV1 + LSTM<br />ResNet101 + Faster R-CNN       | Yes<br />No  | 53.21<br />73.52 |    16.7<br />4.8     |    7.19<br />1.39    |
|     Baseline      |       Single-Frame        |                        ResNet101+RFCN                        |     Yes      |      70.32       |         5.96         |         1.35         |
| Tracking-Assisted |         ECCDetect         | ResNet101 + RFCN (Fixed Rate)<br />ResNet101 + RFCN (Adaptive Rate) | Yes<br />Yes | 63.63<br />62.15 | **40.48<br />59.88** | **10.11<br />14.47** |



## Misc

Code has been trained under:

- Linux with 4 NVIDIA Corporation GV100GL [Tesla V100 SXM2 32GB]

Code has been tested under:

- Linux with 1 NVIDIA  GeForce RTX 2080 [GeForce RTX 2080 11GB]
- Linux with 1 NVIDIA Volta™ architecture with 512 NVIDIA CUDA cores [Jetson AGX Xavier]

