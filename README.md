# Enhanced_lightweight_learning_in_ISAC
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
5. Run the preprocessing scripts CSV_process.py and gen_data_seq.py in order

## Training model:

-- run train_augmt.py to train model. 

-- key parameters
1) args.augment_data: True for enabling data augmentation; False otherwise
2) args.label_smoothing: True for enabling label smoothing; False otherwise
3) args.soft_label_weight: Loss weight \(\lambda\) for soft labels
4) args.beam_soft_label_temperature: Temperature for soflt labels

## Testing model
All trained model along with the hyparameters are under the folder: All_models/

-- run test_model_image.py to test the model

### Models and hyperparameters:
Nine models contained: 
1) Teacher_noAtten.pth: Best teacher model without attention mechanism
2) Teacher_withAtten.pth: Best teacher model with attention mechanism
3) Teacher_selfKD.pth: Best teacher model (including attention mechanism) with self-KD refinement
4) StudentL8_noKD.pth: Student model without KD for input sequence length 8
5) StudentL8_KD.pth: Student model with KD for input sequence length 8
6) StudentL5_noKD.pth: Student model without KD for input sequence length 5
7) StudentL5_KD.pth: Student model with KD for input sequence length 5
8) StudentL3_noKD.pth: Student model without KD for input sequence length 3
9) StudentL3_KD.pth: Student model with KD for input sequence length 3
   
The hyperparameters are shown in the txt files.
