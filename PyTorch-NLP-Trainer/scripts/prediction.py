#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load a pre-trained Chinese BERT model.
# Consider using one better suited for classical texts (e.g., 'hfl/chinese-roberta-wwm-ext' or 'junnyu/wobert-chinese-base')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()

def is_punctuation(char):
    """
    Check if the character is a common Chinese punctuation mark.
    Modify the punctuation list as needed.
    """
    punctuations = "，。！？；：（）【】《》、‘’“”——……"
    return char in punctuations

def predict_missing(text, context_window=128):
    """
    Predict missing characters in text where missing characters are marked by '□'
    for single character gaps and '〼' for multi-character gaps.
    The function processes text in chunks to provide sufficient context for the model.
    """
    # Define topk value for predictions
    topk = 5
    
    # Split the text into overlapping chunks.
    chunks = []
    for i in range(0, len(text), context_window):
        chunk = text[max(0, i-50):min(len(text), i+context_window)]
        chunks.append(chunk)
    
    restored_chunks = []
    for chunk in chunks:
        processed_chunk = chunk
        # First pass: replace single missing characters represented by □ (but not those followed by 〼)
        single_missing_positions = [m.start() for m in re.finditer(r'□(?!〼)', processed_chunk)]
        for pos in single_missing_positions:
            context_before = processed_chunk[:pos]
            context_after = processed_chunk[pos+1:]
            
            # Create a masked version of the text.
            masked_text = context_before + tokenizer.mask_token + context_after
            inputs = tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True)
            
            # Find the position of the [MASK] token.
            mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
            with torch.no_grad():
                outputs = model(**inputs).logits
            
            topk = 5
            probs, indices = torch.topk(outputs[0, mask_position], topk)
            candidates = tokenizer.convert_ids_to_tokens(indices)
            
            # Select the first candidate that is not punctuation.
            predicted_char = None
            for cand in candidates:
                candidate = cand[2:] if cand.startswith('##') else cand
                if not is_punctuation(candidate):
                    predicted_char = candidate
                    break
            
            # Fallback: if all candidates are punctuation, choose the top candidate.
            if predicted_char is None:
                predicted_char = candidates[0][2:] if candidates[0].startswith('##') else candidates[0]
            
            processed_chunk = context_before + predicted_char + context_after
        
        # Second pass: handle multi-character missing sequences marked with 〼.
        while '〼' in processed_chunk:
            match = re.search(r'(.{0,20})〼(.{0,20})', processed_chunk)
            if not match:
                break
            context_before = match.group(1)
            context_after = match.group(2)
            
            # Predict iteratively for a default of 3 characters.
            current_prediction = ""
            for _ in range(3):
                masked_text = context_before + current_prediction + tokenizer.mask_token + context_after
                inputs = tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True)
                
                mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
                with torch.no_grad():
                    outputs = model(**inputs).logits
                
                probs, indices = torch.topk(outputs[0, mask_position], topk)
                candidates = tokenizer.convert_ids_to_tokens(indices)
                
                next_char = None
                for cand in candidates:
                    candidate = cand[2:] if cand.startswith('##') else cand
                    if not is_punctuation(candidate):
                        next_char = candidate
                        break
                if next_char is None:
                    next_char = candidates[0][2:] if candidates[0].startswith('##') else candidates[0]
                
                current_prediction += next_char
                # Optionally, add domain-specific stopping criteria here.
            
            processed_chunk = processed_chunk.replace('〼', current_prediction, 1)
        
        restored_chunks.append(processed_chunk)
    
    # Merge chunks, removing duplicate overlapping parts.
    final_text = restored_chunks[0]
    for i in range(1, len(restored_chunks)):
        overlap = find_overlap(final_text, restored_chunks[i])
        if overlap > 0:
            final_text += restored_chunks[i][overlap:]
        else:
            final_text += restored_chunks[i]
    
    return final_text

def find_overlap(text1, text2, min_overlap=20):
    """
    Find the overlap between the end of text1 and the beginning of text2.
    """
    max_overlap = min(len(text1), len(text2), 100)
    for i in range(max_overlap, min_overlap - 1, -1):
        if text1[-i:] == text2[:i]:
            return i
    return 0

def process_file(input_path, output_path):
    """
    Read text from input_path, process it to predict missing characters, and save the restored text to output_path.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Optionally, split the text into sections based on the full-width period.
        sections = re.split(r'(?<=。)\s+', content)
        
        restored_sections = []
        for section in sections:
            if section.strip():
                restored = predict_missing(section.strip())
                restored_sections.append(restored)
        
        restored_text = '\n'.join(restored_sections)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(restored_text)
            
        print(f"Restoration complete! Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    process_file('medical_text.txt', 'restored_text.txt')
