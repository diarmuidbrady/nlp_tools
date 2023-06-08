import re

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