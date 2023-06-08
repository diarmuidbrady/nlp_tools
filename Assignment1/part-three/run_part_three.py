from naive_bayes import *

print('Please enter a training file (eg. train.pkl)')
train_file = input()

print('Please enter a test file with labels (eg. test.pkl)')
test_file = input()

print('Okay, here we go!')

nb = run_naive_bayes(train_file, test_file)

model_vars = {1:nb.test_reviews, 2:nb.predictions, 3:nb.probs_positive, 4:nb.probs_negative, 5:nb.labels}

print("""Would you like to inspect any of the following variables:
1. test reviews (not recommended if the file is large)
2. predictions
3. positive log probabilities
4. negative log probabilities
5. labels 

('y' or 'n')
""")
a = input()
while a == 'y':
    print('What variable would you like to inspect? (Choose a number between 1 and 5 from above)')
    try:
        n = int(input())
    except:
        print("You didn't enter a number, please try again")
        continue

    print(model_vars[n].values)
    print("Would you like to inspect other variables? ('y' or 'n')")
    a = input()

print("Okay, do you want to predict using your own test file? ('y' or 'n')")
a = input()
if a == 'y':
    print('Enter the file name with the test cases:')
    test_file = input()
    nb = run_test_file(test_file, nb)
    print("""Would you like to inspect any of the following variables:
    1. test reviews (not recommended if the file is large)
    2. predictions
    3. positive log probabilities
    4. negative log probabilities

    ('y' or 'n')
    """)
    model_vars = {1:nb.test_reviews, 2:nb.predictions, 3:nb.probs_positive, 4:nb.probs_negative}
    a = input()
    while a == 'y':
        print('What variable would you like to inspect? (Choose a number between 1 and 4 from above)')
        try:
            n = int(input())
        except:
            print("You didn't enter a number, please try again")
            continue

        print(model_vars[n].values)
        print("Would you like to inspect other variables? ('y' or 'n')")
        a = input()
print("That's all for now! Goodbye :)")