# MTSClean

## Description
This is the experiment code for the MTSClean: "Efficient Constraint-based Cleaning for Multi-Dimensional Time Series Data".
In addition, it also includes some constraint mining algorithms for time series data, such as RFDiscover, DAFDiscover and tsdd

## Requirements
- numpy
- pandas
- scipy
- deap
- pykalman
- multiprocessing
- tkinter
- tqdm

## Running
1. Remove the timestamp from the data file and convert it to a csv file.
2. run
```shell
python ./experiments/comparison.py
```