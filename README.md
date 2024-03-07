# pcml

[![Travis](https://img.shields.io/travis/projectclarify/pcml.svg)](https://travis-ci.org/projectclarify/pcml)

Models and data generators toward developing the capacity to improve users' ability to self-regulate cognitive and emotional states.

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)

View these docs at [docs.cl4rify.org](https://docs.cl4rify.org).

### Objectives and general strategy

At the highest level, the above is to be accomplished by way of a feedback loop whereby a user's state is inferred from one or more modalities and a feedback action is chosen to aid in this task (e.g. play a tone when the user is looking stressed or distracted). For the time being this repository focuses on the state modeling aspect of this (excluding action selection). An intermediate waypoint of the project is to attain the ability to perform beyond state-of-the-art emotion and cognitive state recognition from video, audio, and neural-sensing modalities.

### Problems / data generators

| Dataset | Modalities | Example problem | Status | Code link |
|---|---|---|---|---|
| [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Video, audio | Learn embeddings that are similar for co-occurring signals | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/data_generators/vox_celeb.py) |
| [FEC](https://storage.googleapis.com/public_release/FEC_dataset.zip) | Image | Learn image embeddings from facial expression similarity data. | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/data_generators/fec.py) |

### Models

| Model | Description | Status | Code link |
|---|---|---|---|
| multi_transformer | An experiment with integrating a heterogeneous collection of co-occurring modalities toward enhancing (1) prediction of facial expression labels, and (2) prediction of future game performance. | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/models/multi_transformer.py) |
| img2img_adv | Adversarial img2img translation model, for aiding development of adversarial training code. | WIP | [here](https://github.com/projectclarify/pcml/blob/master/pcml/models/img2img_adv.py) |
