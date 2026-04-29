# ASTRA-sim
[ASTRA-sim](https://astra-sim.github.io/) is a distributed AI system simulator. It models the end-to-end software and hardware stack of modern AI systems - encompassing workload scheduling, collective communication algorithms, and hardware architectures (compute/memory/network). Through a suite of APIs, it enables plug-and-play of external open/proprietary components for modeling different parts of the AI system. This provides end-to-end multi-fidelity simulation capabilities for aiding in design and deployment of next-generation distributed AI systems. 


### Overview and Documentation
Here is a concise visual summary of ASTRA-sim, showing its layers and APIs:
![alt text](https://github.com/astra-sim/astra-sim/blob/master/docs/images/astrasim_overview_codesign.png)

For a comprehensive understanding of the tool, and to gain insights into its capabilities, please visit our [website](https://astra-sim.github.io/).

For information on how to use ASTRA-sim, please visit our [Wiki](https://astra-sim.github.io/astra-sim-docs/index.html).

ASTRA-sim accepts MLCommons Chakra Execution Traces as workload-layer inputs. For details, please visit [Chakra Github](https://github.com/mlcommons/chakra).


### Releases and Contributions

ASTRA-sim is currently at **version 2.0.**
The previous version, ASTRA-sim 1.0, is available in the `ASTRA-sim-1.0` [branch](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0).

We encourage community contributions to ASTRA-sim via PRs.


## Installation Instructions
Install the dependencies:

```
sudo apt -y update
sudo apt -y install coreutils wget vim git
sudo apt -y install gcc-11 g++-11 make cmake 
sudo apt -y install clang-format 
sudo apt -y install libboost-dev libboost-program-options-dev
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install openmpi-bin openmpi-doc libopenmpi-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```
```
cd astra-sim
bash ./install.sh
```
## Test Chakra and AstraSim
```
source astra-sim/astraenv/bin/activate
```
```
chakra_converter Text  --input ./examples/text_converter/text_workloads/Resnet50_DataParallel.txt  --output ./examples/text_converter/text_workloads/Resnet50_DataParallel  --num-npus 8  --num-passes 1
```
```
./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware --workload-configuration=./examples/text_converter/text_workloads/Resnet50_DataParallel  --system-configuration=./examples/text_converter/system.json  --network-configuration=./examples/text_converter/network.yml --remote-memory-configuration=./examples/text_converter/remote_memory.json
```

## Contact Us
For any questions about using ASTRA-sim, you can email the ASTRA-sim User Mailing List: astrasim-users@googlegroups.com

To join the mailing list, please fill out the following form: https://forms.gle/18KVS99SG3k9CGXm6


We appreciate your interest and support in ASTRA-sim!
