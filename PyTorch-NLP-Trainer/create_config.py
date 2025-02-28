import yaml

config = {
    'data_type': 'text_dataset',
    'train_data': 'f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/data/text/data',
    'test_data': 'f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/data/text/data',
    'vocab_file': 'f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/data/text/vocabulary.json',
    'net_type': 'TextCNN',
    'loss_type': 'CrossEntropyLoss',
    'context_size': 32,  # Increased for better feature extraction
    'batch_size': 32,
    'num_epochs': 100,
    'gpu_id': [0],
    'work_dir': 'work_space',
    'flag': 'train',
    'topk': [1, 5],
    'class_name': None,
    'resample': False,
    'num_workers': 4,
    'finetune': False,
    'optim_type': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'milestones': [30, 60, 90],
    'log_freq': 10,
    'kernel_size': [2, 3, 4],  # Multiple kernel sizes for better feature capture
    'embed_size': 300,  # Increased embedding dimension
    'num_channels': 100,  # Adjusted number of channels
    'dropout': 0.5  # Added dropout for regularization
}

with open('f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/configs/config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)