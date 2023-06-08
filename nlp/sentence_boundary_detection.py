# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/1_1_Sentence_Boundary_Detection.ipynb.

# %% auto 0
__all__ = ['sentence_splitter']

# %% ../nbs/1_1_Sentence_Boundary_Detection.ipynb 3
import re

# %% ../nbs/1_1_Sentence_Boundary_Detection.ipynb 4
def sentence_splitter(text:str # The input text to be split
                     ):
    '''
    This sentence splitter takes in a string of text 's' and 
    returns a split version where every sentence is printed on a new line
    '''
    abbr = ['[MDJ]r', 'Hon', 'Esq', 'Prof', 'Mrs','Ms']
    pattern = re.compile(r"(?<!{}.)(?<=[.!?]) ".format('.)(?<!'.join(abbr)))
    sentences = re.split(pattern, text)
    return '\n'.join(sentences)