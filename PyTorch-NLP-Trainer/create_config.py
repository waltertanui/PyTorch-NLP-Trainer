import yaml
import os
import json
import jieba
import csv
import glob
from collections import Counter

# Original config creation
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

# Add functionality to find missing words
def load_vocabulary(vocab_file):
    """Load vocabulary from file"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'vocabulary' in data:
            return set(data['vocabulary'])
        return set(data)

def is_medical_term(word):
    """Check if a word is likely a medical term"""
    medical_keywords = [
        '病', '症', '医', '药', '治疗', '诊断', '患者', '医生', '医院',
        '手术', '疾病', '药物', '处方', '检查', '化验', '血', '心', '肺',
        '肝', '肾', '脑', '骨', '皮肤', '神经', '内科', '外科', '儿科',
        '妇科', '眼科', '耳鼻喉', '口腔', '精神', '肿瘤', '感染', '炎症',
        '疼痛', '发热', '咳嗽', '呼吸', '消化', '循环', '代谢', '免疫'
    ]
    
    # Check if word contains any medical keyword
    for keyword in medical_keywords:
        if keyword in word:
            return True
    return False

def find_missing_words(data_dir, vocab_set, output_csv, medical_only=True):
    """Find words in text files that are not in vocabulary"""
    missing_words = []
    
    # Get all text files recursively
    text_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    # Process each file
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Tokenize with jieba
            words = jieba.lcut(text)
            
            # Find missing words
            for i, word in enumerate(words):
                if word not in vocab_set and len(word.strip()) > 0:
                    # Skip non-medical terms if medical_only is True
                    if medical_only and not is_medical_term(word):
                        continue
                        
                    # Get context (5 words before and after)
                    start_idx = max(0, i-5)
                    end_idx = min(len(words), i+6)
                    context = ' '.join(words[start_idx:i] + ['[' + word + ']'] + words[i+1:end_idx])
                    
                    missing_words.append({
                        'file': file_path,
                        'missing_word': word,
                        'position': i,
                        'context': context
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write results to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'missing_word', 'position', 'context'])
        writer.writeheader()
        writer.writerows(missing_words)
    
    # Print only the missing words (simplified output)
    print("Missing words in Tianhui Medical Simplified:")
    word_counter = Counter([item['missing_word'] for item in missing_words])
    for word, count in word_counter.most_common():
        print(f"{word}")
    
    return missing_words

# Execute the missing words finder
if __name__ == "__main__":
    vocab_file = config['vocab_file']
    data_dir = config['train_data']
    output_csv = 'f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/missing_medical_words.csv'
    
    print("Loading vocabulary...")
    vocab_set = load_vocabulary(vocab_file)
    print(f"Vocabulary size: {len(vocab_set)}")
    
    print(f"Scanning text files in {data_dir} for medical terms...")
    missing_words = find_missing_words(data_dir, vocab_set, output_csv, medical_only=True)
    
    print(f"Results saved to {output_csv}")