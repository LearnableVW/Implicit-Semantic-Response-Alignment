# office-home pda
python run_partial_new.py --s 0 --t 1 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/0_1.txt
python run_partial_new.py --s 0 --t 2 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/0_2.txt
python run_partial_new.py --s 0 --t 3 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/0_3.txt
python run_partial_new.py --s 1 --t 0 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/1_0.txt
python run_partial_new.py --s 1 --t 2 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/1_2.txt
python run_partial_new.py --s 1 --t 3 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/1_3.txt
python run_partial_new.py --s 2 --t 0 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/2_0.txt
python run_partial_new.py --s 2 --t 1 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/2_1.txt
python run_partial_new.py --s 2 --t 3 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/2_3.txt
python run_partial_new.py --s 3 --t 0 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/3_0.txt
python run_partial_new.py --s 3 --t 1 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/3_1.txt
python run_partial_new.py --s 3 --t 2 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --fdim 512 --edim 256 --seed 2019 > results/officehome/3_2.txt

# imagenet-caltech pda
python run_partial_new.py --s 0 --t 1 --dset imagenet_caltech  --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0  --rcwt 1. --alwt 1. --fdim 256 --edim 64 > results/imagenet_caltech/0_1.txt
python run_partial_new.py --s 1 --t 0 --dset imagenet_caltech  --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0  --rcwt 1. --alwt 1. --fdim 2048 --edim 64 > results/imagenet_caltech/1_0.txt
