import os

# Create data directory if it doesn't exist
data_dir = 'f:/Downloads/PyTorch-NLP-Trainer/PyTorch-NLP-Trainer/data/text/data'
os.makedirs(data_dir, exist_ok=True)

# Sample Chinese text
text = """人工智能正在快速发展。
机器学习是人工智能的一个重要分支。
深度学习让计算机具有了强大的能力。
自然语言处理技术日益成熟。
计算机视觉在各个领域得到广泛应用。"""

# Write the text file with UTF-8 encoding
with open(os.path.join(data_dir, 'sample.txt'), 'w', encoding='utf-8') as f:
    f.write(text)