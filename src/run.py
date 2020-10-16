#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:08:58 2020

@author: Bosko
"""


import os
import argparse
import extended_tfidf
import fullbatch_tfidf
import tfidf_tira
import tira_sep_lang 

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Predict task.')
   bar = parser.add_mutually_exclusive_group(required=True)
   bar.add_argument('--mono', action = 'store_const', dest="model", const="tira_separate", help='TIRA monolingual model')    
   bar.add_argument('--multi', action = 'store_const', dest="model", const="tira_multi", help='TIRA multilingual model')    
   bar.add_argument('--fullbatch', action = 'store_const', dest="model", const="fullbatch", help='Monolingual Fullbatch model')    
   bar.add_argument('--extended', action = 'store_const', dest="model", const="extended", help='Monolingual extended grid model')    

   args = parser.parse_args()   
   print(args.model)
   model_key = args.model[0]
   model = {
           'extended' : extended_tfidf.export(),            
           'fullbatch' : fullbatch_tfidf.export(),
           'tira_multi' : tfidf_tira.export(),
           'tira_separate' : tira_sep_lang.export()
            }
   if model_key not in model:
       print("ERROR MODEL")
       exit(1)

