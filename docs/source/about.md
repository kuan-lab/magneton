# About 

**Magneton** is a neuron segmentation pipeline developed by [Kuan Lab](https://www.kuanlab.org/). This pipeline employs a chunked based mode for processing large-scale 3D EM data in neuron segmentation tasks. It is currently available as a CLI tool, supporting both local jobs and HPC jobs.

> Based on Linux machines with NVIDIA GPUs

##### Magneton consists of 3 main modules:

1. Data pre- and post-processing toolkit
- Resize a tif data
- Split a big tif data to blocks   
- Merge h5 files as a tif volume
- Convert a tif/h5 file as a precomputed data
- Downsample a precomputed data
- Generate a mask for affinity map
- Mask a tif/precomputed data

2. Deep learning based affinity maps inference
> Based on [pytorch connectomics](https://connectomics.readthedocs.io/en/latest/index.html).This is a deep learning framework for automatic and semi-automatic annotation of connectomics datasets, powered by [pytorch](https://pytorch.org/).
- Pre-tain / fine-tune a DL model 
- Inference affinity map by blocks 

3. Instance segmentation for affinity maps
> Support global segmentation and large-volume chunking parallel processing, 2D supervoxel and 3D supervoxel 
- Affinity map blocks instance segmentation  
- Aggregation of each blocked segmentation result 

