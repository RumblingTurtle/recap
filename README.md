# ReCAP

Motion capture retargeting library for humanoid robots

## Features
- Easy to use inerface for solving IK for humanoid robots
- MJCF model editor for adding new reference frames to the robot model
- CMU and AMASS mocap datasets support. Library can be extended to support other datasets
- Optimization based SMPLH parameter estimation
- Torch based forward kinematics for MJCF models

# References
- [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) - Extensive collection of MJCF robot models
- [pink](https://github.com/stephane-caron/pink) - Awesome library for diffrential IK 
- [quaternion](https://github.com/moble/quaternion) - Convenient library for quaternion operations
- [SMPL](https://smpl.is.tue.mpg.de) - Vertex based linear human body model
- [AMASS](https://amass.is.tue.mpg.de/) - Aggregated motion capture dataset utilizing SMPL(XH) models
- [CMU](http://mocap.cs.cmu.edu/) - CMU motion capture dataset

## Installation
Dependencies are listed in [`pyproject.toml`](pyproject.toml). However numpy-quaternion needs to be forced to be compiled. For more info refer to the [numpy-quaternion README](https://github.com/moble/quaternion).

```bash
python -m pip install --upgrade --force-reinstall numpy-quaternion
```
Before installing the package you might want to create a new conda environment:

```bash
conda create -n recap python=3.10
```

Then activate the environment:

```bash
conda activate recap
```

Clone the repository:

```bash
git clone https://github.com/RumblingTurtle/recap.git
cd recap
```

Install the package:

```bash
pip install .
```

Or if you are planning to extend the library, you can install the package locally using pip:

```bash
pip install -e .
```

# Contributing

Feel free to contribute to the library by opening a pull request or an issue. 

# Documentation

* [Downloading datasets](docs/downloading_data.md)
* [Adding new datasets](docs/adding_new_datasets.md)
* [Adding new robot models](docs/adding_new_robot_models.md)
* [Solving IK](docs/solving_ik.md)
* [Examples](examples/)