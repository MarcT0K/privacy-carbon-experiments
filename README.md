# Measuring the Carbon Footprint of Cryptographic Privacy-Enhancing Technologies

Authors: Marc Damie, Mihai Pop, and Merijn Posthuma

## Abstract

Privacy-enhancing technologies (PETs) have gained significant attention in response to regulations like the GDPR, driving the development of applications that prioritize user data protection. At the same time, the information and communication technology (ICT) sector faces growing pressure to reduce its environmental footprint, particularly its carbon emissions. While numerous studies have assessed the energy footprint of various ICT applications, the environmental footprint of cryptographic PETs remains largely unexplored.

Our work addresses this gap by proposing a standardized methodology for evaluating the carbon footprint of PETs. We then measure the energy and carbon footprint increase induced by five cryptographic PETs (compared to their non-private equivalent): HTTPS web browsing, encrypted machine learning inference, encrypted ML training, encrypted databases, and email encryption. Our findings reveal significant variability in carbon footprint increases, ranging from a modest twofold increase in HTTPS web browsing to a 10,000-fold increase in encrypted ML.

Our study provides essential data to help decision-makers assess privacy-carbon trade-offs in such applications. Finally, we outline key research directions for developing PETs that balance strong privacy protection with environmental sustainability.

## Building an experiment

To extend this repository with new application, you need to create a new folder with an `experiment.py` file and a `setup.sh`.

The first file will launch all the experiments and handle the carbon footprint measurement using codecarbon.

The second file installs everything necessary to the experiment.

Each experiment should be self-standing and should not use elements from the other folders. If needed, we could create a utility library to gather code snippets reused in multiple experiments.
