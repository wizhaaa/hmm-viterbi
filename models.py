# Name(s): Will Zhang
# Netid(s): wz282
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
from collections import defaultdict
from nltk import classify
from nltk import download
from nltk import pos_tag
import numpy as np

class HMM: 

  def __init__(self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func): 
    """
    Initializes HMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
      vocab: List[String], dataset vocabulary
      all_tags: List[String], all possible NER tags 
      k_t: Float, add-k parameter to smooth transition probabilities
      k_e: Float, add-k parameter to smooth emission probabilities
      k_s: Float, add-k parameter to smooth starting state probabilities
      smoothing_func: (Float, Dict<key Tuple[String, String] : value Float>, List[String]) ->
      Dict<key Tuple[String, String] : value Float> 
    """
    self.documents = documents
    self.labels = labels
    self.vocab = vocab
    self.all_tags = all_tags
    self.k_t = k_t
    self.k_e = k_e
    self.k_s = k_s
    self.smoothing_func = smoothing_func
    self.emission_matrix = self.build_emission_matrix()
    self.transition_matrix = self.build_transition_matrix()
    self.start_state_probs = self.get_start_state_probs()


  def build_transition_matrix(self):
    """
    Returns the transition probabilities as a dictionary mapping all possible
    (tag_{i-1}, tag_i) tuple pairs to their corresponding smoothed 
    log probabilities: log[P(tag_i | tag_{i-1})]. 
    
    Note: Consider all possible tags. This consists of everything in 'all_tags', 
    but also 'qf' our end token. Use the `smoothing_func` and `k_t` fields to 
    perform smoothing.

    Note: The final state "qf" can only be transitioned into, there should be no 
    transitions from 'qf' to any other tag in your matrix

    Output: 
      transition_matrix: Dict<key Tuple[String, String] : value Float>
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    # Initialize a dictionary to store transition counts
    transition_counts = {}
    
    # Iterate through each sentence and its corresponding labels
    for label_sequence in self.labels:
        # Add the start state 'qs' to the beginning of the sequence
        label_sequence = ['qs'] + label_sequence + ['qf']
        
        # Iterate through the sequence and count transitions
        for i in range(1, len(label_sequence)):
            prev_tag = label_sequence[i-1]
            current_tag = label_sequence[i]
            
            if (prev_tag, current_tag) in transition_counts:
                transition_counts[(prev_tag, current_tag)] += 1
            else:
                transition_counts[(prev_tag, current_tag)] = 1
    
    smoothed_transitions = {}
    for prev_tag in self.all_tags + ['qs']:
        for current_tag in self.all_tags + ['qf']:
            smoothed_transitions[(prev_tag, current_tag)] = transition_counts.get((prev_tag, current_tag), 0) + self.k_t
    
    total_counts = {}
    for (prev_tag, current_tag), count in smoothed_transitions.items():
        if prev_tag in total_counts:
            total_counts[prev_tag] += count
        else:
            total_counts[prev_tag] = count
    
    transition_matrix = {}
    for (prev_tag, current_tag), count in smoothed_transitions.items():
        prob = count / total_counts[prev_tag]
        transition_matrix[(prev_tag, current_tag)] = np.log(prob)
    
    return transition_matrix
      


  def build_emission_matrix(self): 
    """
    Returns the emission probabilities as a dictionary, mapping all possible 
    (tag, token) tuple pairs to their corresponding smoothed log probabilities: 
    log[P(token | tag)]. 
    
    Note: Consider all possible tokens from the list `vocab` and all tags from 
    the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.

    Note: The final state "qf" is final, as such, there should be no emissions from 'qf' 
    to any token in your matrix (this includes a special end token!). This means the tag 
    'qf' should not have any emissions, and thus not appear in your emission matrix.
  
    Output:
      emission_matrix: Dict<key Tuple[String, String] : value Float>
      Its size should be len(vocab) * len(all_tags).
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    emission_counts = {}
    
    for sentence, label_sequence in zip(self.documents, self.labels):
        for token, tag in zip(sentence, label_sequence):
            if (tag, token) in emission_counts:
                emission_counts[(tag, token)] += 1
            else:
                emission_counts[(tag, token)] = 1
    
    smoothed_emissions = {}
    for tag in self.all_tags:
        for token in self.vocab:
            smoothed_emissions[(tag, token)] = emission_counts.get((tag, token), 0) + self.k_e
    
    total_counts = {}
    for (tag, token), count in smoothed_emissions.items():
        if tag in total_counts:
            total_counts[tag] += count
        else:
            total_counts[tag] = count
    
    emission_matrix = {}
    for (tag, token), count in smoothed_emissions.items():
        prob = count / total_counts[tag]
        emission_matrix[(tag, token)] = np.log(prob)
    
    return emission_matrix


  def get_start_state_probs(self):
    """
    Returns the starting state probabilities as a dictionary, mapping all possible 
    tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
    parameter to manually perform smoothing.
    
    Note: Do NOT use the `smoothing_func` function within this method since 
    `smoothing_func` is designed to smooth state-observation counts. Manually
    implement smoothing here.

    Note: The final state "qf" can only be transitioned into, as such, there should be no 
    transitions from 'qf' to any token in your matrix. This means the tag 'qf' should 
    not be able to start a sequence, and thus not appear in your start state probs.

    Output: 
      start_state_probs: Dict<key String : value Float>
    """
    # YOUR CODE HERE 
    # raise NotImplementedError()
    start_counts = {}
    for label_sequence in self.labels:
        first_tag = label_sequence[0]
        
        if first_tag in start_counts:
            start_counts[first_tag] += 1
        else:
            start_counts[first_tag] = 1
    
    smoothed_start_counts = {}
    for tag in self.all_tags:
        smoothed_start_counts[tag] = start_counts.get(tag, 0) + self.k_s
    
    total_count = sum(smoothed_start_counts.values())
  
    start_state_probs = {}
    for tag, count in smoothed_start_counts.items():
        prob = count / total_count
        start_state_probs[tag] = np.log(prob)
    
    return start_state_probs


  def get_tag_likelihood(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the tag likelihood used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.
    
    For HMM, this would be the sum of the smoothed log emission probabilities and 
    log transition probabilities: 
    log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].
    
    Note: Treat unseen tokens as an <unk> token.
    
    Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?
    
    Note: Make sure to handle the case where predicted_tag is 'qf'. This corresponds to predicting the last token for a sequence. 
    We can transition into this tag, but (as per our emission matrix spec), there should be no emissions leaving. 
    As such, our probability when predicted_tag = 'qf' should merely be log[P(predicted_tag | previous_tag))].
  
    Input: 
      predicted_tag: String, predicted tag for token at index `i` in `document`
      previous_tag: String, previous tag for token at index `i` - 1
      document: List[String]
      i: Int, index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 
    # raise NotImplementedError()
    token = document[i] if i < len(document) and document[i] in self.vocab else "<unk>"
    
    if i == 0:
        transition_prob = self.start_state_probs.get(predicted_tag, -float('inf'))
    else:
        transition_prob = self.transition_matrix.get((previous_tag, predicted_tag), -float('inf'))
    
    if predicted_tag == 'qf':
        emission_prob = 0.0
    else:
        emission_prob = self.emission_matrix.get((predicted_tag, token), -float('inf'))

    result = transition_prob + emission_prob
    return result
