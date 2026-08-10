# Multi Objective PDBBind
![image](/figure/Multi-Objective%20Optimization%20for%20Binding%20Free%20Energy%20Model%20Figure%201.jpg)
This is the repository for Physics-Guided Rule-Based Graph Convolutional Network(PG-RGCN) Model that uses Multi Objective Optimization models such as PCGrad and GCond to assist in drug discovery. PG-RGCN represents molecular complexes as graphs with atoms as nodes and covelent bonds as edges and the introduction of multi objective loss functions.

##  Installation
Included is the Conda Yml used to install all required dependencies for the Repository via the following:
```
conda create --file pytorch-env.yml
```

## Usage

**PDBbind Dataset**
1. First Run the PCgrad-grid-search-pdbbind.py script to search for ideal physics weights for training. 

2. Run Preato-Front-With-table-new.py searches for the Preato Front of the Physics weights that had been searbed for in the grid search. Make sure to modify line 44 to run the desired subset.
```
 # CHANGE THIS to switch dataset size: 100, 250, 500, 1000, 1500, 2000, 2660
SUBSET_SIZE = 100
```
3. (Optional) Configure the subset size in pdbbind_three_way_testing.py to match the subset size in step 1.
```
    SPLIT_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets/pdbbind_subset_100.pkl'
```
4. Run pdbbind_three_way_testing.py this will train using the physics weights.

**HostGuest Dataset**
1. Run grid_search_w_normailization.py
2. Run pareto-front-hostguest.py to get the knee of the graph
3. Run test-all-with-normalization.py

## Dataset
This repository uses the PDBind Refined v.2015 and HostGuest Systems Dataset.

## References
Juypter Notebooks can be found in the Examples folder. Start from QuickStart-Part1.ipynb.
