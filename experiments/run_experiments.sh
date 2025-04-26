#!/bin/bash

# example of running main experiments in bulk:

# do not forget to run command (giving executable right to user): chmod +x run_experiments.sh 
# then ./run_experiments.sh

python src/gru-ie.py -cfn=conf_gru_ie_1_1_1.yaml -dev=cpu
python src/gru-attn-iec.py -cfn=conf_gru_attn_iec_1_1_1.yaml -dev=cpu

