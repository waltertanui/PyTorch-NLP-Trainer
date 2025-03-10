import csv
import os
from collections import Counter

def print_missing_words(csv_file, show_predictions=True, show_context=False):
    """
    Print missing words from the CSV file in a clean terminal format
    
    Args:
        csv_file: Path to the CSV file with missing words
        show_predictions: Whether to show predictions alongside missing words
        show_context: Whether to show context for each word
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Count occurrences of each missing word
    missing_counter = Counter([row['missing_word'] for row in rows])
    
    # Print header
    print("\n" + "=" * 60)
    print(f"MISSING WORDS FROM TIANHUI MEDICAL SIMPLIFIED")
    print("=" * 60)
    
    # Print each unique missing word with count
    for word, count in missing_counter.most_common():
        if show_predictions:
            # Find the most common prediction for this word
            predictions = [row['predicted_word'] for row in rows if row['missing_word'] == word]
            pred_counter = Counter(predictions)
            top_pred = pred_counter.most_common(1)[0][0]
            
            # Calculate accuracy for this word
            correct = sum(1 for row in rows if row['missing_word'] == word and row['predicted_word'] == word)
            accuracy = correct / count * 100 if count > 0 else 0
            
            print(f"{word:<15} ({count:>3}) → {top_pred:<15} (Accuracy: {accuracy:.1f}%)")
        else:
            print(f"{word:<15} ({count:>3})")
        
        # Show context for the first occurrence if requested
        if show_context:
            for row in rows:
                if row['missing_word'] == word:
                    print(f"    Context: {row['context']}")
                    break
    
    print("-" * 60)
    print(f"Total unique missing words: {len(missing_counter)}")
    print(f"Total missing word instances: {len(rows)}")
    
    # Print prediction accuracy if available
    if show_predictions:
        correct = sum(1 for row in rows if row['is_correct'] == 'True')
        accuracy = correct / len(rows) * 100 if len(rows) > 0 else 0
        print(f"Overall prediction accuracy: {accuracy:.2f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    csv_file = "f:\\Downloads\\PyTorch-NLP-Trainer\\PyTorch-NLP-Trainer\\predicted_medical_words.csv"
    print_missing_words(csv_file, show_predictions=True, show_context=False)