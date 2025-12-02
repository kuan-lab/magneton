# Instance Segmentation
After installing conda, we need to configure Magneton environment.
> Based on Linux machines with NVIDIA GPUs

#### Create a Virtual Environment

```bash
conda create -y -n magneton python=3.9
conda activate magneton
```

Install pytorch with right cuda version (for H100 or A100: usually 12.4):


```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### Install Magneton

##### Install All Modules
We have made modifications based on pytc (*version 0.1*). This section covers the basics of how to download and install this version. 
```bash
git clone https://github.com/kuan-lab/magneton.git
cd magneton
pip install --editable .

```

##### Install Affinity Maps Inference Module
```bash
cd pytorch_connectomics
pip install --editable .
```

##### Install Waterz

```bash
cd ../waterz
pip install --editable .
```

If error: "waterz/backend/types.hpp:3:10: fatal error: boost/multi_array.hpp: No such file or directory"
```bash
conda install boost
```

If error: ModuleNotFoundError: No module named 'Cython'
```bash
pip install cython
```

If error: 'PyDataType_ELSIZE' was not declared in this scope
```bash
pip install --upgrade numpy
```

If error: monai 1.4.0 requires numpy<2.0, >=1.24
After waterz installed
```bash
pip install numpy==1.26.4
```
 
If error: no module named "mahotas"
```bash
pip install mahotas
```

If error: ImportError: ... libstdc++.so.6: version `GLIBCXX_3.4.32' not found ..
```bash
conda install -c conda-forge libstdcxx-ng
```


