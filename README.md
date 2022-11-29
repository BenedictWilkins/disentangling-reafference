# Disentangling Reafferent Effects by Doing Nothing

Supplementary material for __Disentangling Reafferent Effects by Doing Nothing__. 

### Abstract 

_An agent's ability to distinguish between sensory effects that are self-caused, and those that are not, is instrumental in the achievement of its goals. This ability is thought to be central to a variety of functions in biological organisms, from perceptual stabilisation and accurate motor control, to higher level cognitive functions such as planning, mirroring and the sense of agency. Although many of these functions are well studied in AI, this important distinction is rarely made explicit and the focus tends to be on the associational relationship between action and sensory effect or success. Toward the development of more general agents, we develop a framework that enables agents to disentangle self-caused and externally caused sensory effects. Informed by relevant models and experiments in robotics, and in the biological and cognitive sciences, we demonstrate the general applicability of this framework through an extensive experimental evaluation over three different environments._

<p align="center">
  <img src="https://user-images.githubusercontent.com/22711383/204519425-cb8217e8-a0f4-447c-adc4-85363c510938.png" />
</p>

 - (a) shows an observation from `Freeway-v0` environment. The agent (a chicken) is attempting to cross a busy road. The chicken can move `forward`, `backward` or remain in place (`null`).
 - (b) shows the ground truth effect of taking the action `forward` (`X' - X`).
 - (c) shows the predicted reafferent (self-caused) effect.
 - (d) shows the predicted exafferent (externally-caused) effect.

### Technical appendix

The technical appendix refered to in the paper is avaliable to <a id="raw-url" href="https://raw.githubusercontent.com/BenedictWilkins/disentangling-reafference/main/technical-appendix.pdf">download</a>. We recommend taking a look as it gives some additional intuative examples, experiments and further details.

### Reproducability instructions

To run experiments it is advised to create a fresh python environment.
This can be done with [anaconda](https://www.anaconda.com/).
After installing anaconda your terminal should have a prefix like - `(base) NAME:DIR`

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

We recommend trying `*GridWorld2D-Example.ipynb` first, as this is least resource intensive and gives some good intuition about what is being learned.

### Data & Models

Data and pre-trained models can be found in the latest [release](https://github.com/BenedictWilkins/disentangling-reafference/releases) as binary files. They need to be downloaded, extracted and placed in their respective directories (see release description for further instructions).

### Reference

If you use this code in any of your research please consider citing the work:

```
@unpublished{wilkins_reafference_2023,
  author = {Wilkins, Benedict and Stathis, Kostas},
  title  = {Disentangling Reafferent Effects by Doing Nothing}
}
```
__CITATION TO BE UPDATED AFTER THE CONFERENCE__

The full paper will become avaliable in the proceedings of AAAI 2023. A preprint is avaliable for attendees of the conference.



