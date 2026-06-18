# Lightweight-Beam-Tracking-ISAC
Simulator for paper ...


## Dataset Preparation

1. Download the project and extract it to your local machine.

2. Download **Scenario 9** and  **Scenario 31** from:  
   https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9
   https://www.deepsense6g.net/scenarios/Scenarios%2030-39/scenario-31

4. Extract the dataset to form the file structure:

```text
dataset/
└── scenario9/
    ├── unit1/
    └── scenario9.csv
└── scenario31/
    ├── unit1/
    └── scenario31_dev.csv
 ```


## Training model:

-- run train_augmt.py to train model. 

-- key parameters
1) args.augment_data: True for enabling data augmentation; False otherwise
2) args.label_smoothing: True for enabling label smoothing; False otherwise
3) args.soft_label_weight: Loss weight \(\lambda\) for soft labels
4) args.beam_soft_label_temperature: Temperature for soflt labels

## Testing model
All trained model along with the hyparameters are under the folder: All_models/

-- run test_model.py to test the model in the folder: All_models/

### Models and hyperparameters:
12 models contained: 

- Proposed model under four cases: Data augmentation (True/Flase) & LabelSmoothing (True/Flase)
- CNN-GRU model under four cases: Data augmentation (True/Flase) & LabelSmoothing (True/Flase)
- ResNet-GRU model under four cases: Data augmentation (True/Flase) & LabelSmoothing (True/Flase)

The hyperparameters are shown in the txt files.
