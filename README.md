# UHO
## Application UHO: Real-time Slovenian captioning for deaf and hard of hearing

UHO uses OpenAI Whisper models that we fine-tuned on Artur 1.0 dataset.
UHO recognizes real-time speech that is being played on speakers/headphones.

This repository includes:  
  - code for preprocessing Artur 1.0 dataset for training: "./prepare_dataset" directory
  - code for training (Singularity container): "./train_models" directory
  - example code for testing models in Jupyter: "./use_models" directory
  - UHO code for Windows OS
  - UHO code for Android OS
