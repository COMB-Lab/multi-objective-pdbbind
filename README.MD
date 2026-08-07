# Multi Objective PDBBind
![image](/figure/Physics-Guided%20Rule-based%20Graph%20Convolutional%20Network%20(PRGCN)%20with%20Multi-Objective%20Loss%20Function.jpg)
This is the repository for Physics-Guided Rule-Based Graph Convolutional Network(PG-RGCN) Model that uses Multi Objective Optimization models such as PCGrad and GCond to assist in drug discovery. PG-RGCN represents molecular complexes as graphs with atoms as nodes and covelent bonds as edges and the introduction of multi objective loss functions.

##  Installation
Included is the Conda Yml used to install all required dependencies for the Repository via the following:
```
conda create --file pytorch-env.yml
```

## Usage
1. First Run the PCgrad-grid-search-pdbbind.py script to search for ideal physics weights for training.
2. Run Preato-Front-With-table-new.py searches for the Preato Front of the Physics weights that had been searbed for in the grid search.

3. (Optional) Configure the subset size in pdbbind_three_way_testing.py used to run.
```
    SPLIT_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets/pdbbind_subset_100.pkl'
```
4. Run pdbbind_three_way_testing.py this will train using the physics weights.
## Dataset
This repository uses the PDBind Dataset.

## References
Juypter Notebooks can be found in the Examples folder. Start from QuickStart-Part1.ipynb.