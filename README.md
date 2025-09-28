# Energy Consumption of TLS, Searchable Encryption and Fully Homomorphic Encryption

Authors: Marc Damie, Mihai Pop, and Merijn Posthuma

## Abstract

Privacy-enhancing technologies (PETs) have attracted significant attention in response to privacy regulations, driving the development of applications that prioritize user data protection. At the same time, the information and communication technology (ICT) sector faces growing pressure to reduce its environmental footprint, particularly its energy consumption. While numerous studies have assessed the energy consumption of ICT applications, the environmental impact of cryptographic PETs remains largely unexplored.

This work investigates this question by measuring the energy consumption increase induced by three PETs compared to their non-private counterparts: TLS, Searchable Encryption, and Fully Homomorphic Encryption (FHE). These technologies were chosen for two reasons. First, they cover different maturity levels---from the widely deployed TLS protocol to the emerging FHE schemes---allowing us to examine the influence of maturity on energy consumption. Second, they each have well-established applications in industry: web browsing, encrypted databases, and privacy-preserving machine learning.

Our results reveal highly variable energy consumption increases, ranging from 2x for TLS to 10x for Searchable Encryption and 100,000x for FHE. Our experiments demonstrate a simple and reproducible methodology, based on existing open-source software, to quantify the energy costs of PETs. They also highlight the wide spectrum of energy demands across technologies, underscoring the importance of further research on sustainable PET design.
  
## Building an experiment

To extend this repository with new application, you need to create a new folder with an `experiment.py` file and a `setup.sh`.

The first file will launch all the experiments and handle the carbon footprint measurement using codecarbon.

The second file installs everything necessary to the experiment.

Each experiment should be self-standing and should not use elements from the other folders. If needed, we could create a utility library to gather code snippets reused in multiple experiments.
