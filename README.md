# MD-simulation-DNMT3A

## About :dna:
Python scripts used for the Molecular Dynamics Simulation presented in "Preferential interaction of DNMT3A subunits containing the R882H cancer mutation leads to dominant changes of flanking sequence effects", doi.org/10.1016/j.jmb.2022.167482.

For the underlying MD data in the publication go to doi.org/10.18419/darus-2463.

## Installation :rocket:
The MD simulation production and analysis scripts were designed for ubuntu based NVIDIA GPU servers using the cuda toolkit (Version 12). Visit 

https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

for installation instructions.

After installation, check for the correct set up by running

```
nvidia-smi
```
in your terminal.

The output should state the CUDA Version and driver version as displayed below

![Example output from nvidia-smi](nvidia-smi.PNG)

Best way to install packages is by using conda. Either install the full anaconda distribution (https://www.anaconda.com/) or the smaller footprint miniconda (https://docs.conda.io/en/latest/miniconda.html).

create a new conda environment (here named 'sim') by running

```
conde create -n sim
```

install the necessary packages by running

```
conda install conda-forge::openmm
conda install conda-forge::mdtraj
conda install conda-forge::pdbfixer
conda install -c conda-forge contact_map
conda install anaconda::openpyxl
```

## Citation :bookmark_tabs:

For the underlying MD data in the publication go to **doi.org/10.18419/darus-3263**.

Please cite **doi.org/10.18419/darus-2463** for the usage of this dataset.

Please cite **doi.org/10.1016/j.jmb.2022.167482** for the publication.


