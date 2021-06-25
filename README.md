# Implicit-Semantic-Response-Alignment
Pytorch implementation for "Implicit Semantic Response Alignment for Partial Domain Adaptation"

# Prerequisites
- python == 3.6.8
- pytorch ==1.1.0
- orchvision == 0.3.0
- numpy, scipy, PIL, argparse, tqdm, pandas

# Framework
![Alt text](framework.png?raw=true "Title")

# Datasets
The datasets are set up with the same data protocol as [PADA](https://github.com/thuml/PADA/tree/master/pytorch/data).
Please download [Office31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) and [ImageNet-Caltech](https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view) datasets and change the path in the image list files (*.txt) in the './data/' directory.

# Running
Run the code for PDA on Office-Home for Task (A -> C)
> python run_partial_new.py --s 0 --t 1 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 

# Acknowledgement
This project is built on the open-source implementation [BA3US](https://github.com/tim-learn/BA3US)