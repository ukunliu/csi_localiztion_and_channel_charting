# CSI based channel charting and localization

## Installation
torch == 1.13.1
geopy == 2.3.0

autogluon == 0.8.2

keras = 2.1.5

scipy == 1.10.1

skikit-learn == 1.2.2

## Structure


├── **checkpoints**

└── **OpenTest**

    ├── __*.py__
    ├── __*.ipynb__
    ├── **results**
    ├── **data**
    └── **libs**
        ├── another.js
        ├── constants.js
        └── index.js

The main folder is composed by **checkpoints** and **OpenTest**, where the former stores trained model, distinguished by their suffix according to differnt experiment settings. 

Under the main folder, there are two types of files: __*.py__ and __*.ipynb__. In principle, the __*.py__ are scripts for model training and stores trained model under **checkpoint** 
directory. In each script, a __suffix__ variable is set for distinguishing different models in different experienments.

Exceptionelly, __channel_charting_tools.py__ and __offline_mining_tools.py__ are two scripts storing functions in channel charting and triplet/pair offline mining.

__*ipynb__ are notebooks for anaylzing the performance of models, and the results are stored under ```OpenTest/result```.

### **OpenTest**
Test on open datasets. The rirectory **OpenTest** contains the **data**, **result** and **libs** for channel charting. 
#### **data**
Specifically, **data** stores the online CSI data in __dichasus.csv__, and prepared triplets in **opendata_triplets_*.npy** and pairs in **opendata_pairs_*.npy**. For triplets, there are easy, hard mined triplets differentiated by suffix **HN**
#### **libs**
Under **libs**, there are files for creating multilable for autogluon (ensemble method), defining network structre (MLP/Conv1D/Triplet/Siamese), network loss (OnlineTripletLoss/TripletLoss/SiameseLoss/FewShotLoss), and training process.
#### **result**
**result** 

## Reference
The original data __DICHASUS-cf0x__ is accessed from lab in University of Stuttgard, based on indoor experiments: 
<https://dichasus.inue.uni-stuttgart.de/datasets/>

Besides, part of the work has refered to their tutorials:
<https://dichasus.inue.uni-stuttgart.de/tutorials/>
