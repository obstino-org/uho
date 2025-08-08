# UHO
## Application UHO: Real-time Slovenian captioning for deaf and hard of hearing

UHO uses OpenAI Whisper models that we fine-tuned on the Artur 1.0 dataset.
UHO recognizes real-time speech that is being played on speakers/headphones or captured by microphone.

This repository includes:  
  - code for preprocessing Artur 1.0 dataset for training: "./prepare_dataset" directory
  - code for training (Singularity container): "./train_models" directory
  - example code for testing models in Jupyter: "./use_models" directory
  - [UHO code for Windows OS](../../tree/windows-1.0-beta)
  - [UHO code for Android OS](../../tree/android-2.0)
  - look at [Releases](../../releases) for:
    - fine-tuned Whisper models
    - Windows and Android app binaries (.exe installer and .apk)
    - application dependencies and assets for use when developing with our code

Additionally, on Hugging Face may find:
  - our [Whisper base](https://huggingface.co/blko/whisper-base-sl-artur-full-ft) and [Whisper tiny](https://huggingface.co/blko/whisper-tiny-sl-artur-full-ft) fine-tuned models;
  - [preprocessed Artur 1.0 dataset](https://huggingface.co/datasets/blko/artur1_0) that was used in training.

*Click to watch* the Windows app in action:  
[![Watch app in action](https://i.ytimg.com/vi/v-E3Q8McxhY/maxresdefault.jpg)](https://youtu.be/v-E3Q8McxhY)
