#!/usr/bin/env bash
# Usage:
# python predictor.py -c "path/to/config.yaml" -m "path/to/model.pth" -v "path/to/vocabulary.json"

#config_file="work_space/TextCNN_CELoss_20230110175514/config.yaml"
#model_file="work_space/TextCNN_CELoss_20230110175514/model/best_model_117_0.9065.pth"
#vocab_file="work_space/TextCNN_CELoss_20230110175514/vocabulary.json"
#python predictor.py -c $config_file -m $model_file -v $vocab_file

python predictor.py \
  -c "work_space/LSTM_CELoss_20230110175804/config.yaml" \
  -m "work_space/LSTM_CELoss_20230110175804/model/best_model_119_0.9088.pth" \
  -v "work_space/LSTM_CELoss_20230110175804/vocabulary.json" \
