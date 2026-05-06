# BMGRN
This repository contains the implementation of the BMGRN Model.

## Environment Setup

```
conda env create -f DUPLEXenvironment.yaml
```

```
conda env create -f Predictionenvironment.yaml
```

# 1. Graph Embedding

1. Activate the graph embedding environment `conda activate DUPLEX`
2. Navigate to `./code/` directory
3. Excute the python `./train_edge/train.py` 


# 2. GRN prediction

1.Activate the graph GRN prediction environment `conda activate Prediction`

### Shell excution

```
bash
chmod +x run_all_datasets.sh

./run_all_datasets.sh

```
### Manual excution
There are 9 datasets to be evaluated, and each dataset script needs to be executed manually.

**example usage for DREAM5net1 datasets**

```
cd Prediction
cd DREAM5_net1_FGRN
python DREAM5net1_FCV.py
```

Finally, check output folder "results" for results. The csv file shows the mean and standard deviation of AUROC, MCC, F1, Recall, Precision  on this dataset.
