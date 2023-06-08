from bigram import *

import sys
print("Do you have your own corpus? ('y' or 'n')")
s = input()
waiting = True
while waiting:
    if s == 'y':
        print("Enter the corpus below one sentence per line. Enter 'finish' to continue")
        s = input()
        corpus = []
        while s != 'finish':
            corpus.append(s)
            s = input()

        print(f'This is your corpus {corpus}')
        # Create a model with the input corpus
        model = BigramModel(corpus)
        waiting = False
    elif s == 'n':
        corpus = ['b c', 'c c', 'c b', 'b b']
        print(f'This is your corpus {corpus}')
        model = BigramModel(corpus)
        waiting = False

    else:
        print("Please enter either 'y' or 'n'")

ctd = True
while ctd:
    print('Enter a string to find the probability:')
    s = input()
    print(f"The probability of {s} is {model.get_string_probability(s)}.")

    # Ask if they want to continue
    print("Would you like to continue? ('y' or 'n')")
    s = input()
    if s == 'y':
        pass
    elif s == 'n':
        ctd = False

    else:
        print("Please enter either 'y' or 'n'")            

print('Goodbye.')