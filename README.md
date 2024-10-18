![LOGO](asset/logo.png)

Reproducible material for **Robust Full Waveform Inversion with deep Hessian
deblurring** <br> 
Alfarhan M., Ravasi M., Chen F., Alkhalifah  T.


## Project structure
This repository is organized as follows:

* :open_file_folder: **deepinvhessian**: python library containing routines for "DeepFWIInvHessian" Full Waveform Inversion Inverse Hessian with Deep Learning;
* :open_file_folder: **data**: folder containing input data;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **asset**: folder containing logo;

## Supplementary files
The data supporting the findings of this work are available from the corresponding author upon reasonable request.

## Notebooks
The following notebooks are provided:

- :orange_book: ``Run_Conventional_FWI.ipynb`` : notebook performing conventional FWI (will be updated later).
- :orange_book: ``Run_FWI_Born.ipynb`` : notebook estimating the inverse Hessian with the migration/demigration approach (will be updated later).
- :orange_book: ``Run_FWI_PSF.ipynb`` : notebook estimating the inverse Hessian with the PSFs approach (will be updated later).
- :orange_book: ``FWI-LBFGS-Scipy.ipynb`` : notebook performing FWI with L-BFGS algorithm from the Scipy implementation (will be updated later).
- :orange_book: ``PlottingNotebook.ipynb`` : notebook reproducing the figures in the paper (for the first report).
- :orange_book: ``Marmousi_exp.ipynb`` : notebook performing FWI with the Barzilai-Borwein method and the proposed approach on Marmousi.
- :orange_book: ``Marmousi_LBFGS.ipynb`` : notebook performing FWI with L-BFGS on Marmousi.
- :orange_book: ``Marmousi_create_figures.ipynb`` : notebook to visualize the results of the Marmousi experiments.
- :orange_book: ``Volve_synthetic_exp.ipynb`` : notebook performing FWI with the Barzilai-Borwein method and the proposed approach on Volve synthetic.
- :orange_book: ``Volve_synthetic_LBFGS.ipynb`` : notebook performing FWI with L-BFGS on Volve synthetic.
- :orange_book: ``Volve_synthetic_create_figures.ipynb`` : notebook to visualize the results of the Volve synthetic experiments.
- :orange_book: ``Volve_exp.ipynb`` : notebook performing FWI with the Barzilai-Borwein method and the proposed approach on Volve.
- :orange_book: ``Volve_LBFGS.ipynb`` : notebook performing FWI with L-BFGS on Volve.
- :orange_book: ``Volve_imaging.ipynb`` : notebook to compute RTM images and extended images for Volve.
- :orange_book: ``Volve_create_figures.ipynb`` : notebook to visualize the results of the Volve experiments.

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



**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 3.90GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite Us
@misc{alfarhan2024robustwaveforminversiondeep,
      title={Robust Full Waveform Inversion with deep Hessian deblurring}, 
      author={Mustafa Alfarhan and Matteo Ravasi and Fuqiang Chen and Tariq Alkhalifah},
      year={2024},
      eprint={2403.17518},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph},
      url={https://arxiv.org/abs/2403.17518}, 
}
