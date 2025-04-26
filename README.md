# dst-images-insitu

This is supplemental code for developing deep learning models on MESWE dataset. 

Folder [data/](data/) contains data from most extreme solar activity periods. Data are in form of .csv files in raw and processed type.

Folder [experiments/](experiments/) contains python files for preprocessing, deep learning models and postprocessing located in [src](experiments/src/) folder. In [configs](experiments/configs) folder are .yaml configuration files for each model. Models are saved into [models](experiments/models/) folder.

After installing requirements from [requirements](requirements.txt) you can run some experiments with:

```bash
python src/gru-ie.py -cfn=conf_gru_ie_1_1_1.yaml -dev=cpu
```

or

```bash
python src/gru-attn-iec.py -cfn=conf_gru_attn_iec_1_1_1.yaml -dev=cpu
```

to run models in bulk, simply add lines from above to [run_experiments.sh](experiments/run_experiments.sh), change versions of different config files as you wish and run it with

```bash
./run_experiments.sh
```

To change hyperparameters modify provided .yaml config file or create new. Purpouse of using .yaml config files is to track down what model used which parameter durning experimenting. There were a lot of them. All the logs from training will appear in logs/ folder

Folder [dst-images-insitu/notebooks/](notebooks/) contains .ipynb files for demonstration purposes. Reader can reproduce figures and tables and also reproduce models training within the notebook. Code is same as in .py scripts, however we did not use notebooks for training models for two reasons: 1. notebooks tends to be messy, 2. most of the time we trained models in bulk - calling python scripts from bash scripts and more at the same time.
