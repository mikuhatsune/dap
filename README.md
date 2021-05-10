# Detection-Aware Pre-training (DAP)

This is the official repository for our CVPR 2021 paper: DAP: Detection-Aware Pre-training with Weak Supervision ([arXiv: 2103.16651](https://arxiv.org/abs/2103.16651)).

The experiments were based on [`maskrcnn-benchmark`](https://github.com/facebookresearch/maskrcnn-benchmark). The workflow of DAP has 4 steps:

1. Obtain a pre-trained classifier.
2. Run the python scripts to generate the pseudo bounding boxes on ImageNet 1K or 22K (1K classes => 1M images, 22K classes => 14M images).
3. Pre-train a detector on ImageNet and pseudo boxes obtained in 2.
4. Transfer the detector weights in 3 to downstream tasks.

## Source files

### Python scripts for generating pseudo boxes

```
code
├── compute_box_imagenet1k.py
└── compute_box_imagenet22k.py
```

These two files implement the Class Activation Map (CAM) based technique to generate the pseudo bounding boxes, given a pre-trained classifier and the ImageNet dataset variant. The result boxes are in numpy array format. To run the scripts:

```bash
# for imagenet1k (1M)
python code/compute_box_imagenet1k.py
# for imagenet22k (14M), use mpirun
mpirun -n 4 python code/compute_box_imagenet22k.py
```

### Configs

The training configuration files for Step 3 & 4 are under the [configs/](configs/) folder.

```
configs
├── imagenet1k
    ├── dap_in1k_faster_rcnn_r50fpn.yaml      # DAP (Step 2) on ImageNet-1K
    ├── in1k_faster_rcnn_r50fpn_coco.yaml     # COCO template
    ├── in1k_faster_rcnn_r50fpn_voc07.yaml    # VOC07 template
    ├── in1k_faster_rcnn_r50fpn_voc0712.yaml  # VOC0712 template
    └── ... (and similarly for ResNet-101)
└── imagenet22k
    └── ... (similarly for ImageNet-22K)
```

### Experiment shell commands

We take ImageNet-1K and ResNet-50 backbone as the example here. The commands for other configs are similar. While I'm using `mpirun` to demonstrate how to launch experiments, depending on your specific training infrastructure, you may want to use the pytorch distributed launcher or other means.

```bash
# DAP Faster RCNN ResNet-50 on ImageNet-1K
mpirun -n 8 python tools/train_net.py --config-file configs/imagenet1k/dap_in1k_faster_rcnn_r50fpn.yaml


# DAP + Downstream VOC07 detection
mpirun -n 8 python tools/train_net.py --config-file configs/imagenet1k/in1k_faster_rcnn_r50fpn_voc07.yaml OUTPUT_DIR "output/in1k_dap_r50/voc07" MODEL.WEIGHT "output/in1k_dap_r50/model_final.pth"

# Baseline: Classification pre-training (CLS) + Downstream VOC07 detection
mpirun -n 8 python tools/train_net.py --config-file configs/imagenet1k/in1k_faster_rcnn_r50fpn_voc07.yaml


# DAP + Downstream COCO detection
mpirun -n 8 python tools/train_net.py --config-file configs/imagenet1k/in1k_faster_rcnn_r50fpn_coco.yaml OUTPUT_DIR "output/in1k_dap_r50/coco" MODEL.WEIGHT "output/in1k_dap_r50/model_final.pth"

# Baseline: Classification pre-training (CLS) + Downstream COCO detection
mpirun -n 8 python tools/train_net.py --config-file configs/imagenet1k/in1k_faster_rcnn_r50fpn_coco.yaml
```

## Datasets

### Demo implementations of ImageNet-1K/22K

The experiments involve custom impl of the ImageNet datasets. In general, this depends on the code base and data format. You can refer to our impl ([code/imagenet_dataset_impl/](code/imagenet_dataset_impl/)) based on converted TSV image files and numpy array boxes as an example.

## Pre-trained Models and Others

[This Google Drive](https://drive.google.com/drive/folders/1Mo1c0s7xdbse24ZhWPXwyQuvHmVNDpz0?usp=sharing) contains the following things you might find useful:

* ImageNet-22K classification pre-trained ResNet-50/101 models
* ImageNet-1K/22K DAP pre-trained ResNet-50/101 models
* JSON annotation files of the VOC and COCO low-data splits used for Figure 3,4
* A Zip **snapshot of the custom `maskrcnn-benchmark` repo** we used there. It has various modifications and detailed implementations of the custom dataset loaders.

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@inproceedings{zhong2021dap,
  title={DAP: Detection-Aware Pre-training with Weak Supervision},
  author={Zhong, Yuanyi and Wang, Jianfeng and Wang, Lijuan and Peng, Jian and Wang, Yu-Xiong and Zhang, Lei},
  booktitle={CVPR},
  year={2021}
}
```

## License

See [LICENSE](LICENSE) for additional details.