# Measuring the carbon footprint of cryptographically-enhanced privacy

The goal of this project is to measure the carbon overhead induced by privacy enhancement.

To do so, we identify applications with optional crypto-based privacy enhancement. The overhead will be then the carbon difference between a deployment with privacy enhancement and a deployment without.

## Building an experiment

To extend this repository with new application, you need to create a new folder with an `experiment.py` file and a `setup.sh`.

The first file will launch all the experiments and handle the carbon footprint measurement using codecarbon.

The second file installs everything necessary to the experiment.

Each experiment should be self-standing and should not use elements from the other folders. If needed, we could create a utility library to gather code snippets reused in multiple experiments.
