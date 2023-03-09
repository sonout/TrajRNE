<!-- ABOUT THE PROJECT -->
<!-- ## About The Project -->
# TrajRNE
The source code for the paper: “Road Representation Learning with Vehicle Trajectories” accepted in PAKDD 2023 by Stefan Schestakov, Paul Heinemeyer and Elena Demidova.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
To install the required packages we recommend using [Conda](https://docs.conda.io/en/latest/). Our used environment can be easily installed with conda.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/...
   ```
2. Install conda environment
   ```sh
   conda config --env --set channel_priority strict
   conda env create -f environment.yml
   ```
3. Activate the environment
   ```sh
   conda activate road-emb
   ```
4. Install [Fast Map Matching](https://fmm-wiki.github.io/docs/installation/) in the environment (Do the steps while in environment) and for MacOS do also [this](https://github.com/cyang-kth/fmm/pull/214)


<!-- USAGE EXAMPLES -->


### Data Preprocessing

1. Download Trajectory Datasets
   - [Porto](https://www.geolink.pt/ecmlpkdd2015-challenge)
   - [San Francisco](https://crawdad.org/epfl/mobility/20090224/)
2. Preprocess trajectory data using the notebook for the specific dataset found in preprocessing/
3. Generate road networks for the cities using the notebook experiments/generate_road_network.ipynb


## Usage

1. Train the models using the notebooks in models/training/
2. Use the evaluation script evaluation/evaluate.sh to evaluate the models
   ```sh
   ./evaluate.sh -m "trajrne" -t "meanspeed" -p ./results/ -c porto
   ```





