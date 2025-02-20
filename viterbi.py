# Name(s):
# Netid(s):
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np

def viterbi(model, observation, tags):
  """
  Returns the model's predicted tag sequence for a particular observation.
  Use `get_tag_likelihood` method to obtain model scores at each iteration.

  Input: 
    model: HMM model
    observation: List[String]
    tags: List[String]
  Output:
    predictions: List[String]
  """
  # YOUR CODE HERE 
  # raise NotImplementedError()
  num_tags = len(tags)
  num_obs = len(observation)
  
  viterbi_table = np.zeros((num_tags, num_obs))
  backpointer_table = np.zeros((num_tags, num_obs), dtype=int)
  
  for s in range(num_tags):
      
      viterbi_table[s, 0] = model.start_state_probs.get(tags[s], -float('inf')) + \
                              model.get_tag_likelihood(tags[s], 'qs', observation, 0)
      backpointer_table[s, 0] = -1 
      
  for t in range(1, num_obs):
      for s in range(num_tags):
          max_prob = -float('inf')
          best_prev_tag = -1
          
          for s_prev in range(num_tags):
              prob = viterbi_table[s_prev, t-1] + \
                      model.get_tag_likelihood(tags[s], tags[s_prev], observation, t)
              
              if prob > max_prob:
                  max_prob = prob
                  best_prev_tag = s_prev
                  
          viterbi_table[s, t] = max_prob
          backpointer_table[s, t] = best_prev_tag
  
  max_final_prob = -float('inf')
  best_final_tag = -1
  
  for s in range(num_tags):
      
      final_prob = viterbi_table[s, num_obs-1] + \
                    model.get_tag_likelihood('qf', tags[s], observation, num_obs-1)
      
      if final_prob > max_final_prob:
          max_final_prob = final_prob
          best_final_tag = s
  
  
  predictions = []
  current_tag = best_final_tag
  
  for t in range(num_obs-1, -1, -1):
      predictions.append(tags[current_tag])
      current_tag = backpointer_table[current_tag, t]
  
  predictions.reverse()
  
  return predictions