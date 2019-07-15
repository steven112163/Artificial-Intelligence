import random
from copy import deepcopy





class Decision_Tree(object):
	"""
	" Decision tree
	"
	" Attributes:
	"	root: root node of this tree.
	"	train: a list containing training data.
	"	test: a list containing testing data.
	"	actualTrain: a list containing actual targets of train set.
	"	actualTest: a list containing actual targets of test set.
	"	relativeRatio: ratio of considered attributes over all attributes.
	"""
	
	
	
	def __init__(self, train, test, maxDepth, minSize, attriBag, relativeRatio):
		"""
		" __init__ method
		"
		" Create a tree based on the given arguments.
		"
		" Args:
		"	train: training data.
		"	test: testing data.
		"	maxDepth: max depth of this tree.
		"	minSize: min size of every node.
		"	attriBag: attribute bagging. 'use' for using it, 'notuse' otherwise.
		"	relativeRatio: ratio of considered attributes over all attributes.
		"""
		self.train, self.test = deepcopy(train), deepcopy(test)
		self.actualTrain = [row[-1] for row in self.train]
		self.actualTest = [row[-1] for row in self.test]
		self.relativeRatio = relativeRatio
		self.root = self.get_split(self.train, attriBag)
		self.split(self.root, maxDepth, minSize, 1, attriBag)
	
	
	
	def split(self, node, maxDepth, minSize, depth, attriBag):
		"""
		" Split method
		"
		" Create children for a node or make terminal.
		"
		" Args:
		"	node: a node that needs to be splitted.
		"	maxDepth: max depth of this tree.
		"	minSize: min size of every node.
		"	depth: current depth of the given node.
		"	attriBag: attribute bagging. 'use' for using it, 'notuse' otherwise. 
		"""
		left, right = node['groups']
		del(node['groups'])
		
		# check for a no split
		if not left or not right:
			node['left'] = node['right'] = self.to_terminal(left + right)
			return
		
		# check for max depth
		if depth >= maxDepth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
			return
		
		# process left child
		if len(left) <= minSize:
			node['left'] = self.to_terminal(left)
		else:
			node['left'] = self.get_split(left, attriBag)
			self.split(node['left'], maxDepth, minSize, depth + 1, attriBag)
		
		# process right child
		if len(right) <= minSize:
			node['right'] = self.to_terminal(right)
		else:
			node['right'] = self.get_split(right, attriBag)
			self.split(node['right'], maxDepth, minSize, depth + 1, attriBag)
	
	
	
	def to_terminal(self, group):
		"""
		" To_terminal method
		"
		" Create a terminal node value.
		"
		" Args:
		"	group: a group of rows in a node.
		"	
		" Returns:
		"	
		"""
		outcomes = [row[-1] for row in group]
		return max(outcomes, key = outcomes.count)
	
	
	
	def get_split(self, dataset, attriBag):
		"""
		" Get_split method
		"
		" Select the best split point for a dataset.
		"
		" Args:
		"	dataset: a list of rows that needs to be splitted.
		"	attriBag: attribute bagging. 'use' for using it, 'notuse' otherwise. 
		"
		" Returns:
		"	a dictionary with keys 'attribute', 'value' and 'groups'.
		"		'attribute': which attribute is used to split.
		"		'value': splitting threshold.
		"		'groups': a list containing left child list and right child list.
		"""
		classLabels = list(set(row[-1] for row in dataset))
		bAttribute, bValue, bScore, bGroups = 999, 999, 999, None
		randomAttributes = []
		if attriBag is 'use':
			randomAttributes = random.sample(range(len(dataset[0])-1), (int)((len(dataset[0]) - 1)*self.relativeRatio))
		else:
			randomAttributes = range(len(dataset[0])-1)
		datasetCopy = deepcopy(dataset)
		for attribute in randomAttributes:
			datasetCopy.sort(key = lambda element : element[attribute])
			for row in range(len(datasetCopy)-1):
				threshold = (datasetCopy[row][attribute] + datasetCopy[row + 1][attribute]) / 2
				groups = self.test_split(attribute, threshold, datasetCopy)
				gini = self.gini_index(groups, classLabels)
				if gini < bScore:
					bAttribute, bValue, bScore, bGroups = attribute, threshold, gini, groups
		return {'attribute':bAttribute, 'value':bValue, 'groups':bGroups}
	
	
	
	def test_split(self, attribute, value, dataset):
		"""
		" Test_split method
		"
		" Split dataset based on an attribute and an attribute value for testing gini index.
		"
		" Args:
		"	attribute: which attribute is used to split.
		"	value: splitting threshold.
		"	dataset: a list of rows that needs to be splitted.
		"
		" Returns:
		"	left child list and right child list.
		"""
		left, right = [], []
		for row in dataset:
			if row[attribute] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right
	
	
	
	def gini_index(self, groups, classLabels):
		"""
		" Gini_index method
		"
		" Calculate the Gini index for a splitted dataset.
		"
		" Args:
		"	groups: a list containing left child list and right child list.
		"	classLabels: a list of targets.
		"
		" Returns:
		"	gini index value.
		"""
		# count all samples at split point
		numInstances = float(sum([len(group) for group in groups]))
		# sum weighted Gini index for each group
		gini = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			for label in classLabels:
				p = [row[-1] for row in group].count(label) / size
				score += p * p
			# weight the group score by its relative size
			gini += (1.0 - score) * (size / numInstances)
		return gini
	
	
	
	def predict(self, node, row):
		"""
		" Predict method
		"
		" Make a prediction with a decision tree.
		"
		" Args:
		"	node: used to predict the given row.
		"	row: a row that needs prediction.
		"
		" Returns:
		"	
		"""
		if row[node['attribute']] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self.predict(node['right'], row)
			else:
				return node['right']
	
	
	
	def get_tree_accuracy(self):
		"""
		" Get_tree_accuracy method
		"
		" Get the accuracy of this tree.
		"
		" Returns:
		"	the accuracy of this tree in percentage.
		"""
		preTrain = []
		for row in self.train:
			preTrain.append(self.predict(self.root, row))
		
		preTest = []
		for row in self.test:
			preTest.append(self.predict(self.root, row))
		
		return [get_accuracy(self.actualTrain, preTrain), get_accuracy(self.actualTest, preTest)]





def cross_validation_split(dataset, kFolds):
	"""
	" Cross_validation_split method
	"
	" Split dataset into k folds.
	"
	" Args:
	"	dataset: a list of rows that needs to be splitted in to k sets.
	"	kFolds: number of sets.
	"
	" Returns:
	"	a splitted dataset.
	"""
	datasetCopy = deepcopy(dataset)
	random.shuffle(datasetCopy)
	return [datasetCopy[i::kFolds] for i in range(kFolds)]



def train_validation_split(dataset, ratio):
	"""
	" Train_validation_split method
	"
	" Split dataset into train set and test set.
	"
	" Args:
	"	dataset: a list of rows that needs to be splitted into train and test sets.
	"	ratio: ratio of train set over dataset.
	"
	" Returns:
	"	train set and test set.
	"""
	datasetCopy = deepcopy(dataset)
	random.shuffle(datasetCopy)
	return datasetCopy[0:(int)(len(datasetCopy)*ratio)], datasetCopy[(int)(len(datasetCopy)*ratio + 1):-1]



def get_accuracy(actual, predictions):
	"""
	" Get_accuracy method
	"
	" Calculate accuracy by given actual list and prediction list.
	"
	" Args:
	"	actual: a list containing actual targets.
	"	prediction: a list containing predicted targets.
	"
	" Returns:
	"	the accuracy in percentage.
	"""
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predictions[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
