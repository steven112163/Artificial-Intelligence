import pandas as pd
from Tree import *
from Forest import *





def kfolds_tree(dataset, maxDepth, minSize, kFolds, relativeRatio):
	"""
	" Kfolds_tree method
	"
	" Do tree k-fold cross-validation.
	"
	" Args:
	"	dataset: a data list.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	kFolds: number of folds.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	folds = cross_validation_split(dataset, kFolds)
	trainScores, validationScores = [], []
	for fold in folds:
		train = list(folds)
		train.remove(fold)
		train = sum(train, [])
		score = Decision_Tree(train, fold, maxDepth, minSize, 'notuse', relativeRatio).get_tree_accuracy()
		trainScores.append(score[0])
		validationScores.append(score[1])
	
	print('%d folds' % kFolds)
	print('Train: %s' % trainScores)
	print('Mean train: %.3f%%' % (sum(trainScores)/float(len(trainScores))))
	print('Validation: %s' % validationScores)
	print('Mean validation: %.3f%%\n' % (sum(validationScores)/float(len(validationScores))))





def kfolds_forest(dataset, maxDepth, minSize, numTree, kFolds, relativeRatio):
	"""
	" Kfolds_forest method
	"
	" Do forest k-fold cross-validation.
	"
	" Args:
	"	dataset: a data list.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	numTree: number of trees in a forest.
	"	kFolds: number of folds.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	folds = cross_validation_split(dataset, kFolds)
	trainScores, validationScores = [], []
	for fold in folds:
		train = list(folds)
		train.remove(fold)
		train = sum(train, [])
		score = Random_Forest(train, fold, maxDepth, minSize, numTree, 'use', relativeRatio).get_forest_accuracy()
		trainScores.append(score[0])
		validationScores.append(score[1])
	
	print('%d folds' % kFolds)
	print('Train: %s' % trainScores)
	print('Mean train: %.3f%%' % (sum(trainScores)/float(len(trainScores))))
	print('Validation: %s' % validationScores)
	print('Mean validation: %.3f%%\n' % (sum(validationScores)/float(len(validationScores))))





def print_tree_forest(dataset, maxDepth, minSize, numTree, ratio, relativeRatio):
	"""
	" Print_tree_forest method
	"
	" Build a tree, an attribute-bagging forest and a nonattribute-bagging forest.
	"
	" Args:
	"	dataset: a data list.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	numTree: number of trees in a forest.
	"	ratio: ratio of training set over all data set.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	train, test = train_validation_split(dataset, ratio)
	score = Decision_Tree(train, test, maxDepth, minSize, 'notuse', relativeRatio).get_tree_accuracy()
	print('Train, Validation: %s' % score)
	score = Random_Forest(train, test, maxDepth, minSize, numTree, 'use', relativeRatio).get_forest_accuracy()
	print('Train, Validation: %s' % score)
	score = Random_Forest(train, test, maxDepth, minSize, numTree, 'notuse', relativeRatio).get_forest_accuracy()
	print('Train, Validation: %s\n' % score)





def change_ratio(dataset, maxDepth, minSize, numTree, relativeRatio):
	"""
	" Change_ratio method
	"
	" Test different ratio.
	"
	" Args:
	"	dataset: a list containing 6 data lists.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	numTree: number of trees in a forest.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	names = ['*** ellipse100', '*** cross200', '*** iris', '*** glass', '*** ionosphere', '*** wine']
	for i in range(len(dataset)):
		print(names[i])
		for ratio in [0.4, 0.5, 0.6, 0.7]:
			print('Ratio: %.1f' % ratio)
			print_tree_forest(dataset[i], maxDepth, minSize, numTree, ratio, relativeRatio)
		print('')





def change_forest_number(dataset, maxDepth, minSize, kFolds, relativeRatio):
	"""
	" Change_forest_number method
	"
	" Test different number of trees in a forest.
	"
	" Args:
	"	dataset: a list containing 6 data lists.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	kFolds: number of folds.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	names = ['*** ellipse100', '*** cross200', '*** iris', '*** glass', '*** ionosphere', '*** wine']
	for i in range(len(dataset)):
		print(names[i])
		for numTree in range(3, 7):
			print('Number of trees: %d' % numTree)
			kfolds_forest(dataset[i], maxDepth, minSize, numTree, kFolds, relativeRatio)
		print('')





def change_relative_ratio(dataset, maxDepth, minSize, numTree, kFolds):
	"""
	" Change_relative_ratio method
	"
	" Test different relative ratio.
	"
	" Args:
	"	dataset: a list containing 6 data lists.
	"	maxDepth: max depth of a tree.
	"	minSize: min size of a node.
	"	numTree: number of trees in a forest.
	"	kFolds: number of folds.
	"""
	names = ['*** ellipse100', '*** cross200', '*** iris', '*** glass', '*** ionosphere', '*** wine']
	for i in range(2, len(dataset)):
		print(names[i])
		for relativeRatio in [0.4, 0.5, 0.6, 0.7]:
			print('Ralative ratio: %.1f' % relativeRatio)
			kfolds_forest(dataset[i], maxDepth, minSize, numTree, kFolds, relativeRatio)
		print('')





def change_max_depth(dataset, minSize, numTree, kFolds, relativeRatio):
	"""
	" Change_max_depth method
	"
	" Test different max depth.
	"
	" Args:
	"	dataset: a list containing 6 data lists.
	"	minSize: min size of a node.
	"	numTree: number of trees in a forest.
	"	kFolds: number of folds.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	names = ['*** ellipse100', '*** cross200', '*** iris', '*** glass', '*** ionosphere', '*** wine']
	for i in range(len(dataset)):
		print(names[i])
		for maxDepth in [5, 10, 15, 20]:
			print('Max depth: %d' % maxDepth)
			kfolds_forest(dataset[i], maxDepth, minSize, numTree, kFolds, relativeRatio)
		print('')





def change_min_size(dataset, maxDepth, numTree, kFolds, relativeRatio):
	"""
	" Change_max_depth method
	"
	" Test different max depth.
	"
	" Args:
	"	dataset: a list containing 6 data lists.
	"	maxDepth: max depth of a tree.
	"	numTree: number of trees in a forest.
	"	kFolds: number of folds.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	names = ['*** ellipse100', '*** cross200', '*** iris', '*** glass', '*** ionosphere', '*** wine']
	for i in range(len(dataset)):
		print(names[i])
		for minSize in [1, 5, 10, 15]:
			print('Min size: %d' % minSize)
			kfolds_forest(dataset[i], maxDepth, minSize, numTree, kFolds, relativeRatio)
		print('')





if __name__ == '__main__':
	maxDepth, minSize, numTree, kFolds, relativeRatio = 10, 5, 11, 4, 0.8
	dataset = []
	
	# Read ellipse100.txt
	data = pd.read_csv('ellipse100.txt', sep = '\s+', header = None, engine = 'python')
	data = data.values.tolist()
	dataset.append(data)
	
	# Read cross200.txt
	data = pd.read_csv('cross200.txt', sep = ',', header = None)
	data = data.values.tolist()
	dataset.append(data)
	
	# Read iris.data
	data = pd.read_csv('iris.data', sep = ',', header = None)
	data = data.values.tolist()
	dataset.append(data)
	
	# Read glass.data
	data = pd.read_csv('glass.data', sep = ',', header = None)
	data = data.values.tolist()
	data = [row[1:] for row in data]
	dataset.append(data)
	
	# Read ionosphere.data
	data = pd.read_csv('ionosphere.data', sep = ',', header = None)
	data = data.values.tolist()
	dataset.append(data)
	
	# Read wine.data
	data = pd.read_csv('wine.data', sep = ',', header = None)
	data = data.values.tolist()
	data = [row[::-1] for row in data]
	dataset.append(data)
	
	#change_ratio(dataset, maxDepth, minSize, numTree, relativeRatio)
	#change_forest_number(dataset, maxDepth, minSize, kFolds, relativeRatio)
	#change_relative_ratio(dataset, maxDepth, minSize, numTree, kFolds)
	#change_max_depth(dataset, minSize, numTree, kFolds, relativeRatio)
	change_min_size(dataset, maxDepth, numTree, kFolds, relativeRatio)
