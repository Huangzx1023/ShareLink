## Introduction
We propose a novel ShareLink model that aims at achieving 
cross-hemispheric representation alignment through a shared expert mechanism while enhancing generalization capabilities. 
The architecture of ShareLink is as followed.
![image](https://github.com/user-attachments/assets/ab4682a7-60ee-46ef-8f03-ef7eac7cce64)


## Experiment Setting
  ShareLink was trained with a batch size of 512 and 0.1 dropout, using the
AdamW optimizer and cross-entropy loss. Hyperparameters include learning rate
{1e-3,1e-4,1e-5,1e-6}, embedding dimensions {16,32,64}, and 62 experts. Training runs for 500 epochs.
We conducted cross-subject experiments on both the SEED and SEED-IV
datasets. 

  In this experiment, we employed the Leave-One-Subject-Out (LOSO)
cross-validation method. Specifically, for both datasets, the EEG signals of a
single subject were used as the test set, while the EEG signals of the remaining subjects were used as the training set. The average accuracy and standard
deviation across all subjects were adopted as the final evaluation metrics.
