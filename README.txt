CS578 - Statistical Machine Learning- Homework 2
By: Andres Bejarano <abejara@cs.purdue.edu>

For running the program type the following command in console:

	python onlinetraining.py


By default, if no arguments are passed then Perceptron is selected with MaxIterations = 10 and Unigrams as feature set.

IMPORTANT: It is required to have the three data set files (training, validating and testing) in the same folder of the python file and their names should also be train.csv, validation.csv and test.csv

The following are examples for running several experiments:
Perceptron, MaxIterations = 10, Feature Set = Unigrams:  python onlinelearning.py -a 1 -i 10 -f 1
Perceptron, MaxIterations = 10, Feature Set = Bigrams:   python onlinelearning.py -a 1 -i 10 -f 2
Perceptron, MaxIterations = 10, Feature Set = Both:      python onlinelearning.py -a 1 -i 10 -f 3

Winnow, MaxIterations = 10, Feature Set = Unigrams:  python onlinelearning.py -a 2 -i 10 -f 1
Winnow, MaxIterations = 10, Feature Set = Bigrams:   python onlinelearning.py -a 2 -i 10 -f 2
Winnow, MaxIterations = 10, Feature Set = Both:      python onlinelearning.py -a 2 -i 10 -f 3

Additional output for each experiment can be found in experiment1.txt, experiment2.txt, experiment3.txt, experiment4.txt, experiment5.txt and experiment6.txt