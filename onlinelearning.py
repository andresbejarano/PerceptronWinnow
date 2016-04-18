import sys
import csv
import math
import numpy
from numpy import array, dot, random

# template.py
# -------
# Andres Bejarano
# <abejara@cs.purdue.edu>


# Define the names of the set files
trainingDataFile = "train.csv"
validatingDataFile = "validation.csv"
testingDataFile = "test.csv"
neutralWordsFile = "neutral.csv"

# The dictionaries
dictionary = {}
neutralDictionary = {}



##	+--------------------------+
##	| Predict a single example |
##	+--------------------------+
def predict_one(weights, input_snippet):
	
	# Initiate the feature vector
	wt = array([0] * len(weights))
	wt *= 0
	
	# Calculate the dot product (learned label)
	for word in input_snippet:
		if word in dictionary:
			wt[dictionary[word]] = 1
	
	y = dot(weights, wt)
	y = 0 if y <= 0 else 1
	return y


##	+------------+
##	| Perceptron |
##	+------------+
def perceptron(set, weights, learningRate=1):
	
	# Initialize error count
	errors = 0
	
	# Initiate the feature vector
	wt = array([0] * len(weights))
	
	# For each entry t in the training data set
	for xt, yt in set:
		
		# Set the respective values to the feature vector
		wt *= 0
		for word in xt:
			wt[dictionary[word]] = 1
		
		# Calculate the dot product (learned label)
		y = dot(weights, wt)
		y = 0 if y <= 0 else 1
		error = yt - y
		
		# If error then update weight vector
		if error != 0:
			errors += 1
			weights += learningRate * error * wt
			
	# Returns the results
	return [weights, errors]


##	+--------+
##	| Winnow |
##	+--------+
def winnow(set, weights, theta, learningRate=1):

    # Initialize error count
	errors = 0
	
	# Initiate the feature vector
	wt = array([0] * len(weights))
	
	# For each entry t in the training data set
	for xt, yt in set:
		
		# Convert label to the winnow specification
		yt = -1 if yt == 0 else 1
		
		# Set the respective values to the feature vector
		wt *= 0
		for word in xt:
			wt[dictionary[word]] = 1
		
		# Calculate the dot product
		wx = dot(weights, wt)
		
		# If error then update weight vector
		if wx < theta and yt == 1:
			errors += 1
			weights += 2.0 * wt
		if wx > theta and yt == -1:
			errors += 1
			weights += 0.5 * wt
			
	# Returns the results
	return [weights, errors]


##	+---------------------+
##	| Parse the arguments |
##	+---------------------+
def parseArgs(args):
	"""Parses arguments vector, looking for switches of the form -key {optional value}.
	For example:
		parseArgs([ 'template.py', '-a', 1, '-i', 10, '-f', 1 ]) = {'-t':1, '-i':10, '-f':1 }"""
	
	args_map = {}
	curkey = None
	for i in xrange(1, len(args)):
		if args[i][0] == '-':
			args_map[args[i]] = True
			curkey = args[i]
		else:
			assert curkey
			args_map[curkey] = args[i]
			curkey = None
	return args_map

	
##	+---------------------+
##	| Validate the inputs |
##	+---------------------+
def validateInput(args):
	args_map = parseArgs(args)
	
	algorithm = 1 # 1: perceptron, 2: winnow
	maxIterations = 10 # the maximum number of iterations. should be a positive integer
	featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both
	learningRate = 1.0 # The learning rate
	randomWeights = 0 # Indicates if weights are initialized with random values 1: random, 0: no random (then 0)
	
	if '-a' in args_map:
		algorithm = int(args_map['-a'])
	if '-i' in args_map:
		maxIterations = int(args_map['-i'])
	if '-f' in args_map:
		featureSet = int(args_map['-f'])
	if '-l' in args_map:
		learningRate = float(args_map['-l'])
	if '-r' in args_map:
		randomWeights = float(args_map['-r'])
	
	assert algorithm in [1, 2]
	assert maxIterations > 0
	assert featureSet in [1, 2, 3]
	assert randomWeights in [0, 1]
	
	return [algorithm, maxIterations, featureSet, learningRate, randomWeights]


##	+-------------------------------------------------------------+
##	| Build the data set from the indicated file and the set type |
##	+-------------------------------------------------------------+
def buildSet(filename, featureSet, useDictionary=1):

	# Initialize the dictionary and the data
	wordCount = 0
	
	# The set to be built
	localSet = []
	nSet = 0
	pSet = 0
	
	# Read the given file
	with open(filename, 'rb') as csvFile:
		
		# Read the file delimited by commas, ignore quote marks
		reader = csv.reader(csvFile, delimiter=',', skipinitialspace=True)
		
		# For each line in the file
		for line in reader:
			
			# Get the sentence from the line and remove grammatical symbols
			sentence = line[0]
			sentence = sentence.replace(',', ' ');
			sentence = sentence.replace('.', ' ');
			sentence = sentence.replace('[', ' ');
			sentence = sentence.replace(']', ' ');
			sentence = sentence.replace(':', ' ');
			sentence = sentence.replace(';', ' ');
			sentence = sentence.replace(' - ', ' ');
			sentence = sentence.replace(' \'', ' ');
			
			# Split sentence into words
			words = str.split(sentence)
			
			# Keep only not neutral words
			words = [x for x in words if not x in neutralDictionary]
			
			# Save the label of the entry (1 for +, 0 for -)
			yt = 1 if line[1] == '+' else 0
			if yt == 1:
				pSet += 1
			
			# Initialize the feature set
			xt = []
			
			# Build unigrams
			if featureSet == 1:
				
				# For each valid word in the sentence
				for word in words:
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
			
			# Build bigrams
			elif featureSet == 2:
				
				n = len(words) - 1
				for i in range(0, n):
					
					# Build the bigram
					word = (words[i], words[i + 1])
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
			
			# Build unigrams-bigrams
			else:
				
				# Add unigrams first
				for word in words:
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
				
				# Now add bigrams
				n = len(words) - 1
				for i in range(0, n):
					
					# Build the bigram
					word = (words[i], words[i + 1])
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
				
			# Add tuple to the set
			if len(xt) > 0:
				localSet.append((xt, yt))
				nSet += 1
	
	# Return the generated set and the counters
	return localSet, nSet, pSet


##	+-----------------------------------------------+
##	| Generates a dictionary with the neutral words |
##	+-----------------------------------------------+
def buildNeutralDictionary(filename):
	with open(filename, 'rb') as csvFile:
		
		# Read the file delimited by commas, ignore quote marks
		reader = csv.reader(csvFile, delimiter=',', skipinitialspace=True)
		
		# For each line in the file
		for line in reader:
			neutralDictionary[line[0]] = 1


# Performs a classification recognition using the given set and weights vector
def experiment(set, weights):

	# Keep count of matches and mismatches
	matches = 0
	mismatches = 0
	truePositives = 0
	trueNegatives = 0
	predictedPositives = 0
	
	# For each entry t in the testing data set
	for xt, yt in set[0]:
		
		# Calculate the dot product
		y = predict_one(weights, xt)
		if y == 1:
			predictedPositives += 1
		
		# Update the respective counter
		if y == yt:
			matches += 1
			if y == 1:
				truePositives += 1
			else:
				trueNegatives += 1
		else:
			mismatches += 1
	
	# Calculate metrics
	accuracy = (float(truePositives) / float(set[2])) * (float(trueNegatives) / float((set[1] - set[2])))
	precision = float(truePositives) / float(predictedPositives)
	recall = float(truePositives) / float(set[2])
	average = (precision + recall) / 2.0
	fScore = (2.0 * precision * recall) / (precision + recall)
	
	# Print results
	print "MATCHES: ", matches
	print "MISMATCHES: ", mismatches
	print "TRUE POSITIVES: ", truePositives
	print "PREDICTED POSITIVES: ", predictedPositives
	print "ACTUAL POSITIVES: ", set[2]
	print "ACCURACY: ", accuracy
	print "PRECISION: ", precision
	print "RECALL: ", recall
	print "AVERAGE: ", average
	print "F-SCORE: ", fScore
	print ""
	
	# Return values
	#return [matches, mismatches, accuracy, precision, recall, average, fScore]


##	+-------------------+
##	| The main function |
##	+-------------------+
def main():

	# +----------------+
	# | READ ARGUMENTS |
	# +----------------+

	# Validate the arguments
	arguments = validateInput(sys.argv)
	algorithm, maxIterations, featureSet, learningRate, randomWeights = arguments
	#print algorithm, maxIterations, featureSet, learningRate, randomWeights
	
	print "+-------------------------------+"
	print "| EXPERIMENT                    |"
	print "+-------------------------------+"
	print ""
	
	# Indicate which algorithm is running
	alg = "PERCEPTRON" if algorithm == 1 else "WINNOW"
	
	# Indicate which feature set is used
	feat = "UNIGRAMS" if featureSet == 1 else "BIGRAMS" if featureSet == 2 else "BOTH"
	
	print "Algorithm:", alg
	print "Max iterations:", maxIterations
	print "Feature set:", feat
	print ""
	
	
	# +--------------------------------+
	# | READ FILES AND BUILD DATA SETS |
	# +--------------------------------+
	
	# Build the neutral words dictionary
	buildNeutralDictionary(neutralWordsFile)
	
	# Build the sets
	training = buildSet(trainingDataFile, featureSet)
	validating = buildSet(validatingDataFile, featureSet, 0)
	testing = buildSet(testingDataFile, featureSet, 0)
	
	
	# +----------+
	# | TRAINING |
	# +----------+
	
	# The total number of words in the dictionary (sample space)
	nWords = len(dictionary)
	
	# Initialize weight vector
	if algorithm == 1:
		
		# Initialize weights for Perceptron
		if randomWeights == 0:
			weights = array([0] * nWords)
		else:
			weights = random.rand(nWords)
	else:
		
		# Initialize weights for winnow
		weights = array([1] * nWords)
	
	print "+-------------------------------+"
	print "| TRAINING WEIGHTS              |"
	print "+-------------------------------+"
	print ""
	
	# Run the selected algorithm the given number of iterations
	for iteration in range(0, maxIterations):
		
		# Run the selected algorithm
		if algorithm == 1:
			weights, errors = perceptron(training[0], weights, learningRate)
		else:
			weights, errors = winnow(training[0], weights, nWords, learningRate)
		
		print alg, "  Iteration # ", iteration, "\tErrors = ", errors
		
		# Stop iterating if no mistakes are found
		if errors == 0: break
	
	print ""
	
	
	# Show performance results for the training set
	print "+-------------------------------+"
	print "| PERFORMANCE FOR TRAINING SET  |"
	print "+-------------------------------+"
	print ""
	experiment(training, weights)
	
	# Show performance results for the validating set
	print "+---------------------------------+"
	print "| PERFORMANCE FOR VALIDATING SET  |"
	print "+---------------------------------+"
	print ""
	experiment(validating, weights)
	
	# Show performance results for the testing set
	print "+------------------------------+"
	print "| PERFORMANCE FOR TESTING SET  |"
	print "+------------------------------+"
	print ""
	experiment(testing, weights)


### Initiate everything
if __name__ == '__main__':
    main()
