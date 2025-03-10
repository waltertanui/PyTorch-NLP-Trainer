import csv
import re
import os

def combine_text_with_predictions(input_file, output_file):
    # Dictionary to store the best predictions for each line
    line_predictions = {}
    
    with open(input_file, 'r', encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            line_num = row['Line_Number']
            original_text = row['Original_Text']
            
            # Skip empty original text
            if not original_text:
                continue
                
            # Initialize the entry for this line if it doesn't exist
            if line_num not in line_predictions:
                line_predictions[line_num] = {
                    'original': original_text,
                    'predictions': []
                }
            
            # Add the prediction
            prediction = {
                'word': row['Predicted_Word'],
                'probability': float(row['Probability']),
                'position': int(row['Missing_Word_Position'])
            }
            line_predictions[line_num]['predictions'].append(prediction)
    
    # Sort the predictions by line number
    sorted_lines = sorted(line_predictions.items(), key=lambda x: int(x[0]))
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line_num, data in sorted_lines:
            # Sort predictions by position and get highest probability for each position
            position_predictions = {}
            for pred in data['predictions']:
                pos = pred['position']
                prob = pred['probability']
                if pos not in position_predictions or prob > position_predictions[pos][1]:
                    position_predictions[pos] = (pred['word'], prob)
            
            # Create completed text by filling in the original with predictions
            completed_text = data['original']
            
            # Replace missing characters with predictions
            for pos, (word, prob) in sorted(position_predictions.items(), reverse=True):
                if word != '[UNK]' and word != '...' and prob > 0.05:  # Filter out low probability and unknown chars
                    # Find the position of the nth missing character
                    missing_chars = ['□', '〼']
                    count = 0
                    for i, c in enumerate(completed_text):
                        if c in missing_chars:
                            count += 1
                            if count == pos:
                                completed_text = completed_text[:i] + word + completed_text[i+1:]
                                break
            
            # Remove punctuation and write to file
            completed_text_no_punct = re.sub(r'[^\w\s]', '', completed_text)
            outfile.write(f"{completed_text_no_punct}\n")
    
    print(f"Combined text saved to {output_file}")

# Define input and output file paths
input_file = r"f:\Downloads\PyTorch-NLP-Trainer\PyTorch-NLP-Trainer\data\text\data\predictions.csv"
output_file = r"f:\Downloads\PyTorch-NLP-Trainer\PyTorch-NLP-Trainer\data\text\data\combined_text.txt"

combine_text_with_predictions(input_file, output_file)