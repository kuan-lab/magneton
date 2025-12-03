from setuptools import setup, find_packages

def read_requirements():
    """Read requirements.txt and return as list."""
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="magneton",
    version="1.0.0",
    author="Kuan Lab",
    author_email="xxx",
    description="Neuron segmentation and pipeline developed by Kuan Lab.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kuan-lab/magneton.git",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "magneton=magneton.main:main",  # CLI entrypoint
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
