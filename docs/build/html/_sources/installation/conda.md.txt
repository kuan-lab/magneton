# Virtual Environment
We recommend using conda for virtual environment configuration.This section covers the basics of how to download and install conda.
#### Download Miniconda or Anaconda

1. **Miniconda**: A minimal installer for Conda.

2. **Anaconda**: A larger distribution that includes many scientific packages.

Choose either based on your needs. Here are the commands to download Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
 
#### Run the Installer

Make the downloaded script executable and run it:


```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```


#### Follow the Installation Prompts

 
- **Read the License Agreement**: Press `Enter` to scroll through.

-  **Accept the License**: Type `yes` when prompted.

-  **Installation Location**: You can accept the default or specify a different path.

-  **Initialize Miniconda**: Choose whether to initialize Miniconda by typing `yes`.

 
#### Activate Conda

 
After installation, you might need to restart your terminal or run:

```bash
source ~/.bashrc
```

#### Verify Installation


To check if Conda is installed correctly, run:


```bash
conda --version
```

 