import os
import json
import unicodedata
import re
from olaverse.utils.downloader import get_model_path

_YORUBA_MODEL_CACHE = {}
_IGBO_MODEL_CACHE = {}

def remove_tones(text):
    """
    Remove tone marks (acute/grave) but keep dot-below (e.g. ọ, ẹ, ṣ).
    """
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn' or ord(c) == 0x0323  # 0x0323 is combining dot below
    )
    return unicodedata.normalize('NFC', filtered)

def strip_all_diacritics(text):
    """
    Remove all diacritics (tones and dot-below).
    """
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn'
    )
    return unicodedata.normalize('NFC', filtered)

def _load_diacritizer_model(path, is_custom=False):
    if not os.path.exists(path):
        if is_custom:
            raise FileNotFoundError(f"Diacritizer model file not found at: {path}")
        return {"candidates": {}, "transitions": {}, "unigrams": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def viterbi_decode(text, model):
    """
    Viterbi decoder over tokens based on bigram transition probabilities.
    """
    candidates_map = model.get("candidates", {})
    transitions = model.get("transitions", {})
    unigrams = model.get("unigrams", {})
    
    # Tokenize text keeping punctuation/spaces separate
    # We match words vs non-words, preserving combining diacritics with their base characters
    tokens = re.findall(r'\w+[\u0300-\u036f]*|[^\w\s]|\s+', text)
    
    # We want to identify indices of word tokens to perform decoding on
    word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^\w+[\u0300-\u036f]*$', t)]
    
    if not word_indices:
        return text
        
    # Viterbi decoding over word tokens
    # DP table: dp[step] = {candidate_word: (log_prob, backpointer_candidate)}
    dp = []
    
    # Initialization step
    first_word_idx = word_indices[0]
    first_token = tokens[first_word_idx]
    first_token_lower = first_token.lower()
    
    candidates = candidates_map.get(first_token_lower, [first_token_lower])
    
    first_dp = {}
    for cand in candidates:
        # Match capitalization of original token
        if first_token[0].isupper():
            cand_formatted = cand.capitalize()
        else:
            cand_formatted = cand
            
        unigram_prob = unigrams.get(cand, -12.0)
        first_dp[cand_formatted] = (unigram_prob, None)
    dp.append(first_dp)
    
    # Recursion steps
    for step_idx in range(1, len(word_indices)):
        prev_word_idx = word_indices[step_idx - 1]
        curr_word_idx = word_indices[step_idx]
        curr_token = tokens[curr_word_idx]
        curr_token_lower = curr_token.lower()
        
        curr_candidates = candidates_map.get(curr_token_lower, [curr_token_lower])
        
        curr_dp = {}
        for cand in curr_candidates:
            # Match capitalization
            if curr_token[0].isupper():
                cand_formatted = cand.capitalize()
            else:
                cand_formatted = cand
                
            # Find best transition from previous step
            best_prob = -float('inf')
            best_prev = None
            
            for prev_cand, (prev_prob, _) in dp[-1].items():
                prev_cand_lower = prev_cand.lower()
                transition_key = f"{prev_cand_lower} {cand}"
                trans_prob = transitions.get(transition_key, -10.0) # default smoothing log prob
                
                total_prob = prev_prob + trans_prob
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev = prev_cand
                    
            curr_dp[cand_formatted] = (best_prob, best_prev)
            
        dp.append(curr_dp)
        
    # Find the best final state
    best_final_prob = -float('inf')
    best_final_cand = None
    
    for cand, (prob, _) in dp[-1].items():
        if prob > best_final_prob:
            best_final_prob = prob
            best_final_cand = cand
            
    # Backtrack
    best_path = [best_final_cand]
    for step_idx in range(len(word_indices) - 1, 0, -1):
        prev_cand = dp[step_idx][best_path[-1]][1]
        best_path.append(prev_cand)
        
    best_path.reverse()
    
    # Reassemble text
    output_tokens = list(tokens)
    for idx, word_idx in enumerate(word_indices):
        output_tokens[word_idx] = best_path[idx]
        
    return "".join(output_tokens)

def diacritize_yoruba(text, model_path=None):
    """
    Restore full tonal and dot-below diacritics in Yoruba text.
    """
    global _YORUBA_MODEL_CACHE
    
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("yoruba_diacritizer.json")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yoruba_diacritizer.json")
            
    if resolved_path in _YORUBA_MODEL_CACHE:
        model = _YORUBA_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _YORUBA_MODEL_CACHE[resolved_path] = model
        
    return viterbi_decode(text, model)

def diacritize_yoruba_dot_below(text, model_path=None):
    """
    Restore only dot-below diacritics in Yoruba text (strip tonal accents).
    """
    full_diacritized = diacritize_yoruba(text, model_path=model_path)
    return remove_tones(full_diacritized)

def diacritize_igbo(text, model_path=None):
    """
    Restore dot-below diacritics in Igbo text.
    """
    global _IGBO_MODEL_CACHE
    
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("igbo_diacritizer.json")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "igbo_diacritizer.json")
            
    if resolved_path in _IGBO_MODEL_CACHE:
        model = _IGBO_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _IGBO_MODEL_CACHE[resolved_path] = model
        
    return viterbi_decode(text, model)
