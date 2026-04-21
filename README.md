# 1D-LW-ResNet: High-Efficiency UAV Detector Based on RF Signals

## Introduction

1d_lw_ResNet is a simplified version of the ResNet network for the task of classifying and identifying RF signals from UAVs. Our work found that the feature distribution obtained along the frequency dimension in the spectrogram is more sparse, so we used a one-dimensional frequency vector as the input to the network, and changed ResNet to a one-dimensional network as well, which greatly reduces the number of parameters, and with the original Compared with the original method, this project guarantees the same accuracy and greatly reduces the number of parameters.

## Requirements

1. Install `python` and `pytorch` (we use `pytorch>=2.4.0` and `python>=3.10`).
2. `pip install -r requirements.txt`

## Run
### Download Processed Data

We used dataset form [DroneRFa](https://jeit.ac.cn/web/data/getData?dataType=Dataset3).

```
Benchmark
├── Data
    ├── D00 / D01 / D10
    │   ├── T0000
    │   ├── T0001
    │   ├── T0010
    │   ├── T0011
    │   ├── T0100
    │   ├── T0101
    │   ├── T0111
    │   ├── T1001
```

For these time-domain signal datasets, we use the STFT transform to convert them into spectrograms.

*Spectrogram of UAV Signal*

<img src="./img/tf1.png" width="200"/> <img src="./img/tf2.png" width="200"/> 

Converting spectrograms to frequency vectors. And we convert ResNet to 1d_lw_ResNet, as shown in Figure.

<img src="./img/resnet unit.png" width="150"/> <img src="./img/1d-lw-resnet18.png" width="200"/> 

*Confusion Matrix for the different order of operations and vectors.*

<img src="./img/log_sum_time_freq3.png" width="500"/>

*Comparison of model accuracy for different distances, as well as different T and
N values.*

<img src="./img/acc.png" width="500"/>

## Project

```
project
├── test_1024: Spectrogram resolution is 1024, trained in ResNet18 and 1d_lw_ResNet18
├── test_STFT_npy: In order to test 'log' and 'sum' order of operations
├── test: Current test procedures
├── matlab: Some scrpit about matlab to process UAV time RF signal
```

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{liu20241d,
  title={1D-LW-ResNet: High-Efficiency UAV Detector Based on RF Signals},
  author={Liu, Zhenkun and Zhang, Zhiyue and Zhang, Yifei and Mu, Junsheng},
  booktitle={International Conference on Information Processing and Network Provisioning},
  pages={260--271},
  year={2024},
  organization={Springer}
}
```
