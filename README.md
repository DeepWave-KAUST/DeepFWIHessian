![LOGO](asset/logo.png)

Reproducible material for **A deep learning-based inverse Hessian for Full Waveform Inversion - Alfarhan M., Ravasi M., Alkhalifah  T.**


## Project structure
This repository is organized as follows:

* :open_file_folder: **pinnslope**: python library containing routines for "PINNslope" seismic data interpolation and local slope estimation with physics informed neural networks;
* :open_file_folder: **data**: folder containing input data and results;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **asset**: folder containing logo;

## Notebooks
The following notebooks are provided:

- :orange_book: ``Run_Conventional_FWI.ipynb`` : notebook performing conventional FWI.
- :orange_book: ``Run_FWI_Born.ipynb`` : notebook estimating the inverse Hessian with the migration/demigration approach.
- :orange_book: ``Run_FWI_PSF.ipynb`` : notebook estimating the inverse Hessian with the PSFs approach.
- :orange_book: ``FWI-LBFGS-Scipy.ipynb`` : notebook performing FWI with L-BFGS algorithm from the Scipy implementation..
- :orange_book: ``PlottingNotebook.ipynb`` : notebook reproducing the figures in the paper.  

## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate deepinvhessian
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```



**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

