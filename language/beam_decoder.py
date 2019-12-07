# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:12:43 2019

@author: belov
"""
import numpy as np
import math

def beam_search_decoder(n, model, encoded_image, word_map, id_to_word_dictionary, max_sentence_length):
    
    sentence_id = np.zeros((n, max_sentence_length))
    
    prob_seq_lst = []
    
    for i in range(n):
      prob_seq_lst.append((0,['<START>']))
      
    i = 0
    
    while i < max_sentence_length:
        
      prob_seq_lst_temp = []
      
      for prev_seq_ind in range(len(prob_seq_lst)):
        word_id = word_map[prob_seq_lst[prev_seq_ind][1][-1]]
        sentence_id[prev_seq_ind, i] = word_id
        next_word_id = model.predict([np.array([encoded_image]), np.reshape(sentence_id[prev_seq_ind, :], (1, max_sentence_length))])
        next_word_id = next_word_id.flatten()
        
        mid_seq = prob_seq_lst[prev_seq_ind][1].copy()
        next_word_id_sorted = np.sort(next_word_id)
        p = []
        c = 1
        while len(p) < n:
            prob = next_word_id_sorted[-c]
            if prob not in p:
                p.append(prob)
                ind = np.where(next_word_id == prob)
                word = id_to_word_dictionary[ind[0][0]]
    
                if c == 1:
                    mid_seq.append(word)
                else:
                    mid_seq[i+1] = word
            
                prob_seq_lst_temp.append((prob_seq_lst[prev_seq_ind][0] + math.log(prob),mid_seq))
                mid_seq = mid_seq.copy()
                
            c += 1
          
      prob_seq_lst = []
      i += 1
      prob_seq_sorted = sorted(prob_seq_lst_temp, reverse=True)
      p = []
      
      for prob_seq in prob_seq_sorted:
          if prob_seq[0] not in p:
              p.append(prob_seq[0])
              prob_seq_lst.append(prob_seq)
          if len(prob_seq_lst) == n:
            break
            
    sent_sorted = sorted(prob_seq_lst, reverse=True)      
    final_output_sentences = [prob_sentence[1] for prob_sentence in sent_sorted]
    
    return final_output_sentences