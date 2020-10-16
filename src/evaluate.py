# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:02:18 2020

@author: Bosko
"""

import argparse
import tfidf_tira
import config
import os

def evaluate(path,path_out):
    path_en = os.path.join(path,'en')
    path_out_en = os.path.join(path_out,'en')
    os.mkdir(path_out_en)
    tfidf_tira.fit(path_en,path_out_en,'en')
    path_es = os.path.join(path,'es')
    path_out_es = os.path.join(path_out,'es')
    os.mkdir(path_out_es)
    tfidf_tira.fit(path_es,path_out_es,'es')
    
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Predict task.')
   parser.add_argument('-i', dest='input_dir', metavar='i', type=str, nargs=1,
                       help='Input directory.',required=True)
   parser.add_argument('-o', dest='output_dir', metavar='o', type=str, nargs=1,
                       help='Output directory.',required=True)    
   args = parser.parse_args()
   print(args.input_dir)
   input_dir = args.input_dir[0]
   output_dir = args.output_dir[0]
   evaluate(input_dir,output_dir)
