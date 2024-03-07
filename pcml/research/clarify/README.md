
# Project Clarify

Models and data generators toward developing the capacity to improve users' ability to self-regulate cognitive and emotional states.

Objectives and general strategy
At the highest level, the above is to be accomplished by way of a feedback loop whereby a user's state is inferred from one or more modalities and a feedback action is chosen to aid in this task (e.g. play a tone when the user is looking stressed or distracted). For the time being this repository focuses on the state modeling aspect of this (excluding action selection). An intermediate waypoint of the project is to attain the ability to perform beyond state-of-the-art emotion and cognitive state recognition from video, audio, and neural-sensing modalities.

Please refer to the [Q1 research context and plan document](https://docs.google.com/document/d/1diHaoMLmFshejwptbtlDMoKgrsW9xjoNPIV-1BO_Jz4/edit) for more details.

### Problems / data generators

| Dataset | Modalities | Example problem | Status | Code link |
|---|---|---|---|---|
| [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Video, audio | Learn embeddings that are similar for co-occurring signals | WIP | [here](https://github.com/reverence-innovation/rvrnc/blob/master/rvrnc/data_generators/vox_celeb.py) |
| [FEC](https://storage.googleapis.com/public_release/FEC_dataset.zip) | Images | Learn embeddings based on facial expression similarity | WIP | [here](https://github.com/reverence-innovation/rvrnc/blob/master/rvrnc/data_generators/fec.py) |


### Models

| Model | Description | Status | Code link |
|---|---|---|---|
| multi_transformer | An experiment with integrating a heterogeneous collection of co-occurring modalities toward enhancing (1) prediction of facial expression labels, and (2) prediction of future game performance. | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/models/multi_transformer.py) |
| img2img_adv | Adversarial img2img translation model, for aiding development of adversarial training code. | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/models/img2img_adv.py) |

