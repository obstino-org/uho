# UHO (!note: older legacy version)
## Application UHO: Real-time Slovenian captioning for deaf and hard of hearing

## IMPORTANT DEPRECATION NOTE: New Version Available

### *This version has been deprecated*

**New and improved UHO Slovene v3** using NeMo ASR models is available on: <https://github.com/obstino-org/UHO_Slovene_v3> (also avaiable on Google Play!)

### Description

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

### Relevant papers:

Related paper:

B. Kovačič, J. Brest, B. Bošković: Sistem za podnaslavljanje slovenskega govora za gluhe in naglušne (available on <https://ev.fe.uni-lj.si/5-2025/Kovacic.pdf>)

Related master's thesis for older UHO version:

B. Kovačič, J. Brest, B. Bošković: Razpoznava slovenskega govora v aplikaciji za osebe z okvaro sluha (available on <https://dk.um.si/IzpisGradiva.php?lang=slv&id=93585>)

You may cite as:
```
@article{kovacic2025podnaslavljanje,
  title={Sistem za podnaslavljanje slovenskega govora za gluhe in naglu\v{s}ne},
  author={Kova\v{c}i\v{c}, Bla\v{z} and Brest, Janez and Bo\v{s}kovi\'{c}, Borko},
  journal={Elektrotehni\v{s}ki Vestnik},
  volume={92},
  number={5},
  year={2025},
  publisher={Elektrotehni\v{s}ki Vestnik}
}
```

and

```
@mastersthesis{Kova\v{c}i\v{c}_2025, place={Maribor}, title={Razpoznava slovenskega govora v aplikaciji za osebe z okvaro sluha : magistrsko delo}, url={https://dk.um.si/IzpisGradiva.php?lang=slv&id=93585}, abstractNote={V okviru magistrskega dela smo razvili aplikacijo za osebe z okvaro sluha, ki z razpoznavanjem slovenskega govora omogo\v{c}a realno\v{c}asovno podnaslavljanje predvajanega govora. Za razpoznavanje govora aplikacija uporablja modele Whisper podjetja OpenAI, ki smo jih dou\v{c}ili na superra\v{c}unalniku VEGA s pomo\v{c}jo korpusa Artur 1.0. Pri tem smo za primerjavo rezultatov u\v{c}ili dva modela razli\v{c}nih velikosti. Za ve\v{c}ji model smo na testni mno\v{z}ici dosegli stopnjo napa\v{c}no razpoznanih besed 11,38 %, medtem ko smo za hitrej\v{s}i, manj\v{s}i model dosegli 15,19 %. Realno\v{c}asovno izvajanje smo zagotovili z razli\v{c}nimi optimizacijami dekodiranja \v{z}etonov in s pomo\v{c}jo ustreznih zaledij za sklepanje z modeli.}, school={Faculty of Electrical Engineering and Computer Science - FERI, University of Maribor}, author={Kova\v{c}i\v{c}, Bla\v{z}}, year={2025}}
```
