# Speaker-Follower Models for Vision-and-Language Navigation

This repository contains the code for the following paper:

* D. Fried*, R. Hu*, V. Cirik*, A. Rohrbach, J. Andreas, L.-P. Morency, T. Berg-Kirkpatrick, K. Saenko, D. Klein**, T. Darrell**, *Speaker-Follower Models for Vision-and-Language Navigation*. in NIPS, 2018. ([PDF](https://arxiv.org/pdf/1806.02724.pdf))
```
@inproceedings{fried2018speaker,
  title={Speaker-Follower Models for Vision-and-Language Navigation},
  author={Fried, Daniel and Hu, Ronghang and Cirik, Volkan and Rohrbach, Anna and Andreas, Jacob and Morency, Louis-Philippe and Berg-Kirkpatrick, Taylor and Saenko, Kate and Klein, Dan and Darrell, Trevor},
  booktitle={Advances in Neural Information Processing Systems (NIPS)},
  year={2018}
}
```
(*, **: indicates equal contribution)

Project Page: http://ronghanghu.com/speaker_follower

## Installation

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install PyTorch following the instructions on https://pytorch.org/ (we used PyTorch 0.3.1 in our experiments).
3. Download this repository or clone **recursively** with Git, and then enter the root directory of the repository:  
```
# Make sure to clone with --recursive
git clone --recursive https://github.com/ronghanghu/speaker_follower.git
cd speaker_follower
```

If you didn't clone with the `--recursive` flag, then you'll need to manually clone the pybind submodule from the top-level directory:
```
git submodule update --init --recursive
```
4. Compile the Matterport3D Simulator:
```
mkdir build && cd build
cmake ..
make
cd ../
```

## Train and evaluate on the Room-to-Room (R2R) dataset

### Download and preprocess the data

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symbol link it to `exp_clevr_snmn/clevr_dataset`. After this step, the file structure should look like
```
exp_clevr_snmn/clevr_dataset/
  images/
    train/
      CLEVR_train_000000.png
      ...
    val/
    test/
  questions/
    CLEVR_train_questions.json
    CLEVR_val_questions.json
    CLEVR_test_questions.json
  ...
```

(Optional) If you want to run any experiments on the CLEVR-Ref dataset for the referential expression grounding task, you can download it from [here](http://people.eecs.berkeley.edu/~ronghang/projects/snmn/CLEVR_loc.tgz), and symbol link it to `exp_clevr_snmn/clevr_loc_dataset`. After this step, the file structure should look like
```
exp_clevr_snmn/clevr_loc_dataset/
  images/
    loc_train/
      CLEVR_loc_train_000000.png
      ...
    loc_val/
    loc_test/
  questions/
    CLEVR_loc_train_questions.json
    CLEVR_loc_val_questions.json
    CLEVR_loc_test_questions.json
  ...
```

2. Extract visual features from the images and store them on the disk. In our experiments, we extract visual features using ResNet-101 C4 block. Then, construct the "expert layout" from ground-truth functional programs, and build image collections (imdb) for CLEVR (and CLEVR-Ref). These procedures can be down as follows.
```
./exp_clevr_snmn/tfmodel/resnet/download_resnet_v1_101.sh  # download ResNet-101

cd ./exp_clevr_snmn/data/
python extract_resnet101_c4.py  # feature extraction
python get_ground_truth_layout.py  # construct expert policy
python build_clevr_imdb.py  # build image collections
cd ../../

# (Optional, if you want to run on the CLEVR-Ref dataset)
cd ./exp_clevr_snmn/data/
python extract_resnet101_c4_loc.py  # feature extraction
python get_ground_truth_layout_loc.py  # construct expert policy
python build_clevr_imdb_loc.py  # build image collections
cd ../../
```

### Training

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Train on the CLEVR dataset for VQA:  
    - with ground-truth layout  
`python exp_clevr_snmn/train_net_vqa.py --cfg exp_clevr_snmn/cfgs/vqa_gt_layout.yaml`  
    - without ground-truth layout  
`python exp_clevr_snmn/train_net_vqa.py --cfg exp_clevr_snmn/cfgs/vqa_scratch.yaml`

2. (Optional) Train on the CLEVR-Ref dataset for the REF task:  
    - with ground-truth layout  
`python exp_clevr_snmn/train_net_loc.py --cfg exp_clevr_snmn/cfgs/loc_gt_layout.yaml`  
    - without ground-truth layout  
`python exp_clevr_snmn/train_net_loc.py --cfg exp_clevr_snmn/cfgs/loc_scratch.yaml`

3. (Optional) Train jointly on the CLEVR and CLEVR-Ref datasets for VQA and REF tasks:  
    - with ground-truth layout  
`python exp_clevr_snmn/train_net_joint.py --cfg exp_clevr_snmn/cfgs/joint_gt_layout.yaml`  
    - without ground-truth layout  
`python exp_clevr_snmn/train_net_joint.py --cfg exp_clevr_snmn/cfgs/joint_scratch.yaml`

Note:
* By default, the above scripts use GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2). During training, the script will write TensorBoard events to `exp_clevr_snmn/tb/{exp_name}/` and save the snapshots under `exp_clevr_snmn/tfmodel/{exp_name}/`.
* When training without ground-truth layout, there is some variance in performance between each run, and training sometimes gets stuck in poor local minima. In our experiments, before evalutating on the test split, we took 4 trials and selected the best one based on validation performance.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate on the CLEVR dataset for the VQA task:  
`python exp_clevr_snmn/test_net_vqa.py --cfg exp_clevr_snmn/cfgs/{exp_name}.yaml TEST.ITER 200000`  
where `{exp_name}` should be one of `vqa_gt_layout`, `vqa_scratch`, `joint_gt_layout` and `joint_scratch`.  
*Expected accuracy: 96.6% for `vqa_gt_layout`, 93.0% for `vqa_scratch`, 96.5% for `joint_gt_layout`, 93.9% for `joint_scratch`.* Note:
    - The above evaluation script will print out the accuracy (only for val split) and also save it under `exp_clevr_snmn/results/{exp_name}/`. It will also save a prediction output file in this directory.  
    - The above evaluation script will generate 100 visualizations by default, and save it under `exp_clevr_snmn/results/{exp_name}/`. You may change the number of visualizations with `TEST.NUM_VIS` parameter (e.g. appending `TEST.NUM_VIS 400` to the commands above to generate 400 visualizations).
    - By default, the above script evaluates on the *validation* split of CLEVR. To evaluate on the *test* split, append `TEST.SPLIT_VQA test` to the command above. As there is no ground-truth answers for *test* split in the downloaded CLEVR data, **the displayed accuracy will be zero on the test split**. You may email the prediction outputs in `exp_clevr_snmn/results/{exp_name}/` to the CLEVR dataset authors for the *test* split accuracy.  
    - By default, the above script uses GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2).  

2. (Optional) Evaluate on the CLEVR-Ref dataset for the REF task:  
`python exp_clevr_snmn/test_net_loc.py --cfg exp_clevr_snmn/cfgs/{exp_name}.yaml TEST.ITER 200000`  
where `{exp_name}` should be one of `loc_gt_layout`, `loc_scratch`, `joint_gt_layout` and `joint_scratch`.  
*Expected accuracy (Precision@1): 96.0% for `loc_gt_layout`, 93.4% for `loc_scratch`, 96.2% for `joint_gt_layout`, 95.4% for `joint_scratch`.* Note:
    - The above evaluation script will print out the accuracy (Precision@1) and also save it under `exp_clevr_snmn/results/{exp_name}/`.  
     - The above evaluation script will generate 100 visualizations by default, and save it under `exp_clevr_snmn/results/{exp_name}/`. You may change the number of visualizations with `TEST.NUM_VIS` parameter (e.g. appending `TEST.NUM_VIS 400` to the commands above to generate 400 visualizations).
    - By default, the above script evaluates on the *validation* split of CLEVR-Ref. To evaluate on the *test* split, append `TEST.SPLIT_LOC loc_test` to the command above.  
    - By default, the above script uses GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2).  

## Train and evaluate on the VQAv1 and VQAv2 datasets

### Download and preprocess the data

1. Download the VQAv1 and VQAv2 dataset annotations from http://www.visualqa.org/download.html, and symbol link them to `exp_vqa/vqa_dataset`. After this step, the file structure should look like
```
exp_vqa/vqa_dataset/
  Questions/
    OpenEnded_mscoco_train2014_questions.json
    OpenEnded_mscoco_val2014_questions.json
    OpenEnded_mscoco_test-dev2015_questions.json
    OpenEnded_mscoco_test2015_questions.json
    v2_OpenEnded_mscoco_train2014_questions.json
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_OpenEnded_mscoco_test-dev2015_questions.jso
    v2_OpenEnded_mscoco_test2015_questions.json
  Annotations/
    mscoco_train2014_annotations.json
    mscoco_val2014_annotations.json
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json
    v2_mscoco_train2014_complementary_pairs.json
    v2_mscoco_val2014_complementary_pairs.json
```

2. Download the COCO images from http://mscoco.org/, and symbol link it to `exp_vqa/coco_dataset`. After this step, the file structure should look like
```
exp_vqa/coco_dataset/
  images/
    train2014/
      COCO_train2014_000000000009.jpg
      ...
    val2014/
    test2015/
  ...
```

3. Extract visual features from the images and store them on the disk. In our experiments, we extract visual features using ResNet-152 C5 block. Then, build image collections (imdb) for VQAv1 and VQAv2. These procedures can be down as follows.

```
./exp_vqa/tfmodel/resnet/download_resnet_v1_152.sh  # Download ResNet-152

cd ./exp_vqa/data/
python extract_resnet152_c5_7x7.py  # feature extraction for all COCO images
python build_vqa_imdb_r152_7x7.py  # build image collections for VQAv1
python build_vqa_imdb_r152_7x7_vqa_v2.py  # build image collections for VQAv2
cd ../../
```
(Note that this repository already contains the "expert layout" from parsing results using Stanford Parser. They are the same as in [N2NMN](http://ronghanghu.com/n2nmn).)

### Training

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Train on the VQAv1 dataset:  
    - with ground-truth layout  
`python exp_vqa/train_net_vqa.py --cfg exp_vqa/cfgs/vqa_v1_gt_layout.yaml`  
    - without ground-truth layout  
`python exp_vqa/train_net_vqa.py --cfg exp_vqa/cfgs/vqa_v1_scratch.yaml`  

2. Train on the VQAv2 dataset:  
    - with ground-truth layout  
`python exp_vqa/train_net_vqa.py --cfg exp_vqa/cfgs/vqa_v2_gt_layout.yaml`  
    - without ground-truth layout  
`python exp_vqa/train_net_vqa.py --cfg exp_vqa/cfgs/vqa_v2_scratch.yaml`  

Note:
* By default, the above scripts use GPU 0, and train on the union of *train2014* and *val2014* splits. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2). During training, the script will write TensorBoard events to `exp_vqa/tb/{exp_name}/` and save the snapshots under `exp_vqa/tfmodel/{exp_name}/`.

### Test

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  

1. Evaluate on the VQAv1 dataset:  
`python exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{exp_name}.yaml TEST.ITER 20000`  
where `{exp_name}` should be one of `vqa_v1_gt_layout` and `vqa_v1_scratch`. Note:
    - By default, the above script evaluates on the *test-dev2015* split of VQAv1. To evaluate on the *test2015* split, append `TEST.SPLIT_VQA test2015` to the command above.
    - By default, the above script uses GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2).  
    - The above evaluation script will not print out the accuracy (**the displayed accuracy will be zero**), but will write the prediction outputs to `exp_vqa/eval_outputs/{exp_name}/`, which can be uploaded to the evaluation sever (http://www.visualqa.org/roe.html) for evaluation. *Expected accuracy: 66.0% for `vqa_v1_gt_layout`, 65.5% for `vqa_v1_scratch`.*  

2. Evaluate on the VQAv2 dataset:  
`python exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{exp_name}.yaml TEST.ITER 40000`  
where `{exp_name}` should be one of `vqa_v2_gt_layout` and `vqa_v2_scratch`. Note:
    - By default, the above script uses GPU 0. To run on a different GPU, append `GPU_ID` parameter to the commands above (e.g. appending `GPU_ID 2` to use GPU 2).  
    - The above evaluation script will not print out the accuracy (**the displayed accuracy will be zero**), but will write the prediction outputs to `exp_vqa/eval_outputs_vqa_v2/{exp_name}/`, which can be uploaded to the evaluation sever (http://www.visualqa.org/roe.html) for evaluation. *Expected accuracy: 64.0% for `vqa_v2_gt_layout`, 64.1% for `vqa_v2_scratch`.*  

## Acknowledgements

This repository is built upon the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) codebase.
