import time
import copy
from multiprocessing import Manager, Pool
from functools import partial



# Macros
DIR_DOWN = 'D'
DIR_RIGHT = 'A'



"""
" A puzzle of multiple lines with unique line ID.
"
"	@param lines: a crossword line in the puzzle.
"""
class Puzzle(object):
	def __init__(self):
		self.lines = {}
	
	"""
	" Add a new crossword line in the puzzle.
	"
	"	@param length: lenth of the word.
	"	@param direction: direction of the line, down or across.
	"	@param intersectionPoints: intersection points of the line.
	"	@param lineId: unique ID of this line.
	"""
	def add_line(self, length, direction, intersectionPoints, lineId):
		newLine = Puzzle.CrosswordLine(length, direction, intersectionPoints)
		self.lines[lineId] = newLine
	
	"""
	" A crossword line in the puzzle.
	"
	" 	@param length: lenth of the word.
	"	@param direction: direction of the line, down or across.
	"	@param intersectionPoints: intersection points of the line.
	"""
	class CrosswordLine(object):
		def __init__(self, length, direction, intersectionPoints):
			self.length = length
			self.direction = direction
			self.intersectionPoints = intersectionPoints
	
	"""
	" An intersection point.
	"
	"	@param firstId: ID of first line.
	"	@param secondId: ID of second line.
	"	@param firstIntersect: intersection point of first line.
	"	@param secondIntersect: intersection point of second line.
	"""
	class IntersectionPoint(object):
		def __init__(self, firstId, secondId, firstIntersect, secondIntersect):
			self.firstId = firstId
			self.secondId = secondId
			self.firstIntersect = firstIntersect
			self.secondIntersect = secondIntersect
		
		"""
		" Check whether letters at the intersection point of two words are the same.
		"
		"	@param firstWord: first word.
		"	@param secondWord: second word.
		"""
		def words_fit(self, firstWord, secondWord):
			return (firstWord[self.firstIntersect] == secondWord[self.secondIntersect])



"""
" Generate a puzzle and find intersection points.
"
"	@param puzzles: list of tuples, each list is a puzzle constraint.
"	@param puzzleNumber: indicate use which puzzle constraint to generate the puzzle.
"	@return puzzle.
"""
def generate_puzzle(puzzles, puzzleNumber):
	# List of tuples, each of the form (x, y, length, direction).
	rows = []
	# Add each row in puzzles into rows.
	# A row in puzzle is (x y length direction), so indexes are 0, 2, 4, 6.
	for i in range(len(puzzles[puzzleNumber])):
		x = int(puzzles[puzzleNumber][i][0])
		y = int(puzzles[puzzleNumber][i][2])
		length = int(puzzles[puzzleNumber][i][4])
		direction = puzzles[puzzleNumber][i][6]
		rows.append((x, y, length, direction))
	
	X_FIELD =  0;
	Y_FIELD = 1;
	LEN_FIELD = 2;
	DIR_FIELD = 3;
	
	puzzle = Puzzle()
	
	# Calculate the intersection points of each row, and combine these points and the row information, by adding it to the puzzle.
	for i, row in enumerate(rows):
		lineId = i
		x = row[X_FIELD]
		y = row[Y_FIELD]
		length = row[LEN_FIELD]
		direction = row[DIR_FIELD]
		intersections = []
		
		print("at row " + str(lineId) + " with x, y, len, dir " + str(x) + ", " + str(y) + ", " + str(length) + ", " + str(direction));
		
        # Check all other lines to see if they intersect with this one.
		for j, oRow in enumerate(rows):
			oLineId = j;
			ox = oRow[X_FIELD]
			oy = oRow[Y_FIELD]
			oLength = oRow[LEN_FIELD]
			oDirection = oRow[DIR_FIELD];
			# Make sure that they are not the same row, or they are parallel.
			if lineId == oLineId or direction == oDirection:
				continue
			
			if direction == DIR_DOWN: # Implying oRow has direction right
				if ox <= x and ox + oLength >= x and y <= oy and y + length >= oy: # If they intersect
					intersect = Puzzle.IntersectionPoint(lineId, oLineId, oy - y, x - ox)
					intersections.append(intersect)
			elif direction == DIR_RIGHT: # Implying oRow has direction down
				if x <= ox and x + length >= ox and oy <= y and oy + oLength >= y: # If they intersect
					intersect = Puzzle.IntersectionPoint(lineId, oLineId, ox - x, y - oy)
					intersections.append(intersect)
		
		puzzle.add_line(length, direction, intersections, lineId)
	
	return puzzle



"""
" Solve the puzzle with given words.
"
"	@param puzzle: puzzle composing of multiple lines.
"	@param words: list of words.
"""
def solve(puzzle, words, wordsByLength):
	# Next, we find the length of each line and take the list of words that can
	# fit there from the word_by_length dictionary
	fittingWords = {}
	for lineId, line in puzzle.lines.items():
		if line.length not in wordsByLength:
			return None
		fittingWords[lineId] = copy.copy(wordsByLength[line.length])
		if fittingWords[lineId] == None:
			return None
	
	keylist = list(puzzle.lines.keys())
	if not keylist:
		return None
	
	# Finally, we pass in the information we have generated to the 
	# find_solutions function, which will modify solution_set to contain all of
	# the solutions to the puzzle, so it can be returned to the user.
	manager = Manager()
	solutionSet = manager.list()
	find_solutions(puzzle, fittingWords, solutionSet)
	
	return solutionSet



"""
" Find all solutions of the puzzle.
"
"	@param
"	@param
"""
def find_solutions(puzzle, fittingWords, solutionSet):
	initialId = get_optimal_guess_line(list(puzzle.lines.keys()), fittingWords)
	currentSolution = {}
	"""
	for possibleWord in fittingWords[initialId]:
		guess_word(possibleWord, puzzle, initialId, fittingWords, currentSolution, solutionSet)
	"""
	pool = Pool(8)
	func = partial(guess_word, puzzle = puzzle, lineId = initialId, fittingWords = fittingWords, currentSolution = currentSolution, solutionSet = solutionSet)
	pool.map(func, fittingWords[initialId])



def guess_word(guess, puzzle, lineId, fittingWords, currentSolution, solutionSet):
	currentSolution[lineId] = guess
	guessedLine = puzzle.lines[lineId]
	# We copy new_fitting_words so that we can remove words from it without
	# modifying the fitting_words list in other solution branches.
	newFittingWords = copy.deepcopy(fittingWords)
	
	# Remove all words from the new_fitting_words list that don't fit with the
	# new guess.
	for intersect in guessedLine.intersectionPoints:
		# If the spot is filled, don't bother doing any calculations
		if intersect.secondId in currentSolution:
			continue
		
		secondIdFittingWords = copy.copy(newFittingWords[intersect.secondId])
		for word in secondIdFittingWords:
			if not intersect.words_fit(guess, word):
				newFittingWords[intersect.secondId].remove(word)
		
		if not newFittingWords[intersect.secondId]:
			del currentSolution[lineId]
			return
	
	solvedIds = currentSolution.keys()
	allIds = puzzle.lines.keys()
	possibleIds = [x for x in allIds if x not in solvedIds]
	
	# If there are no more id's to guess, meaning all of the lines have been
	# filled in, we have found a solution, so we add it to the current_solutions
	# list.
	if not possibleIds:
		solutionSet.append(copy.copy(currentSolution))
		del currentSolution[lineId]
		return
	
	targetId = get_optimal_guess_line(possibleIds, newFittingWords)
	
	for possibleWord in newFittingWords[targetId]:
		if possibleWord not in currentSolution.values():
			guess_word(possibleWord, puzzle, targetId, newFittingWords, currentSolution, solutionSet)
	
	# It is important to remove the guess from the current_solutions list after
	# we are done with it so that it is not still there when we are attempting
	# to start a new search branch.
	del currentSolution[lineId]



def get_optimal_guess_line(idList, fittingWords):
	if not idList:
		return None
	
	targetId = idList[0]
	lowestPossibilityCount = len(fittingWords[targetId])
	for lineId in idList:
		numFittingWords = len(fittingWords[lineId])
		if numFittingWords < lowestPossibilityCount:
			lowestPossibilityCount = numFittingWords
			targetId = lineId
	
	return targetId



def main():
	# Four-pair width and height of four puzzles
	width = [4, 7, 6, 8]
	height = [5, 7, 5, 8]
	
	# Read word file
	wordFile = open("English words 3000.txt", "r")
	words = wordFile.read().split("\n")
	wordFile.close()
	
	# Read puzzle file
	puzzles = [line.rstrip('\n').split("   ") for line in open("puzzle.txt", "r")]
	
	# Generate puzzle
	print("Generating puzzle...")
	puzzle = generate_puzzle(puzzles, 1)
	
	# Cluster words by each word's length
	wordsByLength = {}
	for word in words:
		length = len(word)
		if length not in wordsByLength:
			wordsByLength[length] = []
		wordsByLength[length].append(word)

	
	print("Solving...")
	startTime = time.time()
	solutions = solve(puzzle, words, wordsByLength)
	seconds = round(time.time() - startTime)

	if seconds == 1:
		secondsEnd = ''
	else:
		secondsEnd = 's'

	print("Solved in {} second{}".format(seconds, secondsEnd))

	if solutions:
		print("Displaying solutions...")
		print("Total solutions: {}".format(len(solutions)))
		# print the solutions
	else:
		print("No solutions found")
	
	
if __name__ == '__main__':
	main()
