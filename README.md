# Reafference

Supplementary code for the paper __Disentangling Reafferent Effects by Doing Nothing__

Reproducability instructions:

To run experiments it is advised to create a fresh python environment.
This can be done with anaconda - https://www.anaconda.com/
After install anaconda your terminal should have a prefix like - `(base) NAME:DIR`

Create a new conda environment:

```
conda create --name reaff python=3.8
conda activate reaff
```

navigate to the locally cloned repository directory containing `setup.py` and install the required dependencies: 

```
cd <PATH>
pip install .
```

This should install the dependencies listed in the `setup.py` file.

Experiments are written in jupyter notebook files, to use the conda kernel and visualise results properly use the following command:
 
```
python -m ipykernel install --user --name=reaff
jupyter nbextension install --user --py widgetsnbextension
jupyter nbextension enable --user --py widgetsnbextension
```

When opening a notebook file, choose the kernel `reaff`


The directory `reafference` contains code that supports the notebooks, including model and environment definitions. Notebook files that contain experiments that are presented in the main body of the paper are marked with an * in their file name.

We recommend trying the GridWorld2D first, as this is least resource intensive and gives some good intuition about what is being learned.

If you use this code in any of your research please consider citing:

(BIBTEX COMING SOON)
