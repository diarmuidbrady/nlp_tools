class BigramModel:
    '''
    The BigramModel class takes a corpus as input. 
    The corpus is a list of sentence strings.
    Each word or token is separated by spaces.
    '''
    def __init__(self, 
                 corpus:list # This is the training corpus for input
                ):

        self.corpus = corpus

        # Create a list of lists representing each sentence
        self.sentences = [sentence.split() for sentence in self.corpus]
        
        # Create a list of tokens from sentences
        self.tokens = [word for sentence in self.sentences for word in sentence]
        
        # bigram_counts contains the counts of each bigram in the corpus
        self.bigram_counts = {}
        
        # word_counts contains the counts
        self.word_counts = {}
        for sentence in self.sentences:
            # start index at 1 and look back one for each bigram
            for i in range(1, len(sentence)):
                prev_word, next_word = sentence[i-1], sentence[i]
                
                # Create a count for the first word in each bigram
                # Check if we have seen this bigram before
                if prev_word in self.word_counts:
                    self.word_counts[prev_word] += 1
                    
                # else we add it to the dictionary
                else:
                    self.word_counts[prev_word] = 1

                bigram = (prev_word, next_word)
                                    
                # Create a count for each bigram in the corpus
                if bigram in self.bigram_counts:
                    self.bigram_counts[bigram] += 1
                else:
                    self.bigram_counts[bigram] = 1
                    
        # bigram_probabilities contains the probabilities of each bigram in the corpus
        self.bigram_probabilities = {}
        for bigram in self.bigram_counts:
            prev_word = bigram[0]
            
            # Given a previous word what is the bigram probability?
            probability = self.bigram_counts[bigram] / self.word_counts[prev_word]
            self.bigram_probabilities[bigram] = probability

    def get_bigram_probability(self,
                               bigram:list # list of 2 strings
                              ):
        '''
        Returns a probability for a given bigram input.
        
        Example:
        model = BigramModel(corpus)
        print(model.get_bigram_probability(['c' 'b']))
        
        >> 0.25
        '''
        
        # Check if the bigram exists else return zero
        if bigram in self.bigram_probabilities:
            return self.bigram_probabilities[bigram]
        else:
            return 0
        
    def get_string_probability(self,
                               string:str # a normal sentence
                              ):
        '''
        Returns the probability of a string of text
        
        Example:
        model = BigramModel(corpus)
        print(model.get_string_probability('c b')
        
        >> 0.25
        '''
        tokens = string.split()
        probability = self.tokens.count(tokens[0]) / len(self.tokens)
        for i in range(1, len(tokens)):
            bigram = (tokens[i-1], tokens[i])
            probability *= self.get_bigram_probability(bigram)
        return probability