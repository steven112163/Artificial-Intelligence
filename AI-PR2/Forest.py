from Tree import *





class Random_Forest(object):
	"""
	" Random forest
	"
	" Attributes:
	"	forest: store all trees in this forest.
	"	train: a list containing training data.
	"	test: a list containing testing data.
	"	actualTrain: a list containing actual targets of train set.
	"	actualTest: a list containing actual targets of test set.
	"""
	
	
	
	def __init__(self, train, test, maxDepth, minSize, numTree, attriBag, relativeRatio):
		"""
		" __init__ method
		"
		" Create a forest based on the given arguments.
		"
		" Args:
		"	train: training data.
		"	test: testing data.
		"	maxDepth: max depth of every tree.
		"	minSize: min size of every node of a tree.
		"	numTree: number of trees in this forest.
		"	attriBag: attribute bagging. 'use' for using it, 'notuse' otherwise.
		"	relativeRatio: ratio of considered attributes over all attributes.
		"""
		self.forest = []
		self.train, self.test = deepcopy(train), deepcopy(test)
		self.actualTrain = [row[-1] for row in self.train]
		self.actualTest = [row[-1] for row in self.test]
		for tree in range(numTree):
			trainSet, testSet = train_validation_split(self.train, 0.8)
			self.forest.append(Decision_Tree(trainSet, testSet, maxDepth, minSize, attriBag, relativeRatio))
	
	
	
	def get_forest_accuracy(self):
		"""
		" Get_forest_accuracy method
		"
		" Get the training and testing accuracy of this forest.
		"
		" Returns:
		"	the accuracy of this forest in percentage.
		"""
		preTrain = []
		for row in self.train:
			prediction = []
			for tree in self.forest:
				prediction.append(tree.predict(tree.root, row))
			preTrain.append(max(prediction, key = prediction.count))
		
		preTest = []
		for row in self.test:
			prediction = []
			for tree in self.forest:
				prediction.append(tree.predict(tree.root, row))
			preTest.append(max(prediction, key = prediction.count))
		
		return [get_accuracy(self.actualTrain, preTrain), get_accuracy(self.actualTest, preTest)]
