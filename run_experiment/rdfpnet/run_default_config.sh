#!/bin/bash -i

SEEDS=(12 22 32 42 52)

DATASETS=('Liver-Disorders' 'Shedden_2008' 'Cervical Cancer' 'Parkinson Dataset' 'Hepatic Encephalopathy')

for SEED in "${SEEDS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        python run_default_config_RDFPNET.py \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --catenc \

        
        python run_default_config_RDFPNET.py \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --catenc \
    done
done