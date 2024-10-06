import numpy as np
import random
import torch

def reset_seed(seed):
    """For deterministic training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_accuracy(logits, targets):
    """Computes the average accuracy."""
    acc = (torch.argmax(logits, dim=1) == targets).float().sum().item() / len(targets)
    return acc

def clean_meta_data(text):
    """For cleaning the meta data of mrqa."""
    has_comma = ',' in text # Check if the text has commas
    # Check if the text has the pattern of single-letter spacing
    if "  " in text:
        phrases = text.split(',')
        corrected_phrases = []
        for phrase in phrases:
            # Split by double spaces
            words = phrase.split("  ")
            # Remove extra spaces from each word
            corrected_words = [''.join(word.split()) for word in words]
            # Rebuild the phrase
            corrected_phrase = '  '.join(corrected_words)
            corrected_phrases.append(corrected_phrase.strip())

        corrected_text = ', '.join(corrected_phrases) if has_comma else ''.join(corrected_phrases)
        return corrected_text
    else:
        # If the text is already correctly formatted, return it as is
        return text

