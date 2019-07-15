import logging
import sys
import copy

# create logger
logger = logging.getLogger('ChessBoard.py')
logger.setLevel(logging.CRITICAL)
# create console handler, print logging at stdout, and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.CRITICAL)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class ChessBoard(object):
    def __init__(self, new_board: bool = False, input_board=None,
                 game_turn: int = 0, mcts_turn: int = 0,
                 is_black: bool = True, is_black_turn: bool = True):
        """
        Constructor of ChessBoard
        :param new_board: Is this ChessBoard new.
        :param input_board: Get this ChessBoard.board from input_board, pass by value
        :param game_turn: How many turns since the game start. When reaches 200, game over.
        :param is_black: Is we are playing as black piece? (MAX is black or white)
        :param is_black_turn: Is this the turn of black moving?
        """
        if input_board is None:
            input_board = [[0]]
        self.board = [[]]
        self.pretty_board = [[]]
        self.game_turn = game_turn
        self.mcts_turn = mcts_turn
        self.is_black = is_black
        self.is_black_turn = is_black_turn
        if new_board:
            self.initialize_board()
        else:
            self.board = copy.deepcopy(input_board)

    def initialize_board(self):
        """
        Clear & Reset this ChessBoard
        :return: None
        """
        logger.debug('Initialize new chess board...')
        self.clear_board()
        self.board[0] = [1, 0, 0, 0, 0, 0, 0, 0]
        self.board[1] = [0, 1, 0, 0, 0, 0, 0, 2]
        self.board[2] = [1, 0, 1, 0, 0, 0, 2, 0]
        self.board[3] = [0, 1, 0, 0, 0, 2, 0, 2]
        self.board[4] = [1, 0, 1, 0, 0, 0, 2, 0]
        self.board[5] = [0, 1, 0, 0, 0, 2, 0, 2]
        self.board[6] = [1, 0, 0, 0, 0, 0, 2, 0]
        self.board[7] = [0, 0, 0, 0, 0, 0, 0, 2]

    def clear_board(self):
        """
        Clear all pieces of this ChessBoard.
        :return: None
        """
        self.board.clear()
        for _ in range(8):
            self.board.append([0] * 8)

    def print_raw(self):
        """
        Print list of list.
        :return: None
        """
        for row in self.board:
            print(row)

    def print_board(self):
        """
        Pretty print the chess board.
        :return: None
        """
        self.pretty_board.clear()

        for row in range(8):
            piece_type = [' ', "\u25CB", "\u25CF"]
            self.pretty_board.append([piece_type[x] for x in self.board[row]])

        print('~  ', end='')
        for col_idx in range(8):
            print(' {}'.format(col_idx), end='')

        print('')
        for row_idx in range(8):
            print('{}  |'.format(row_idx), end='')
            for piece in self.pretty_board[row_idx]:
                print(piece, end='|')
            print('')
        print('')

    def put_piece(self, location: tuple = (0, 0), piece: int = 0):
        """
        Put a piece on this ChessBoard, WILL change this ChessBoard
        :param location: two digit tuple, (row, col) zero based
        :param piece: color of piece, 0:None, 1:Black, 2:White
        :return: None
        """

        try:
            row = location[0]
            col = location[1]
            if piece == 0:
                self.board[row][col] = 0
            elif piece == 1:
                self.board[row][col] = 1
            elif piece == 2:
                self.board[row][col] = 2
            else:
                self.board[row][col] = 0
                logger.warning('Undefined piece type: {}'.format(piece))
        except IndexError:
            logger.error('Index {} is out of range or undefined.'.format(location))
            return

    def move_piece(self, steps: list):
        """
        This method will return a list[list]. Will 'not' change this ChessBoard.board
        Also will check if the move is valid, including 'capture' the opponent.
        If invalid move found during process, will return current legal steps.
        :param steps: list of tuples, [(0, 1), (0, 3), (2, 3)]
        means move the piece at (0,1) to (0,3) then (2,3)
        :return: list[list]
        """
        new_board = ChessBoard(input_board=self.board,
                               game_turn=self.game_turn + 1,
                               is_black=self.is_black,
                               is_black_turn=not self.is_black_turn)
        piece_type = self.board[steps[0][0]][steps[0][1]]
        is_jumping = False

        if len(steps) == 1:
            return new_board

        for index, coordinate in enumerate(steps):
            # Starting location
            if index == 0:
                if piece_type == 0:
                    logger.warning('No piece at location {}'.format(steps[0]))
                    return new_board
                else:
                    logger.info('Moving {} piece at {}'.format(self.get_piece_string(piece_type), steps[0]))
                    continue
            # Follow up moving coordination
            else:
                row_diff = abs(coordinate[0] - steps[index - 1][0])
                col_diff = abs(coordinate[1] - steps[index - 1][1])
                if index == 1:
                    if row_diff > 1 or col_diff > 1:
                        is_jumping = True

                if new_board.board[coordinate[0]][coordinate[1]] != 0:
                    logger.error('Illegal move! The coordinate {} is occupied.'.format(coordinate))
                    return new_board

                # Adjacency move
                elif is_jumping is False:
                    if index > 1:
                        logger.error('Illegal move! More than one adjacency move.')
                        return new_board
                    if row_diff == 1 and col_diff == 0:
                        logger.info('Vertical step, from {} to {}.'.format(steps[index - 1], coordinate))
                        new_board.put_piece(steps[index - 1], 0)
                        new_board.put_piece(coordinate, piece_type)
                    elif row_diff == 0 and col_diff == 1:
                        logger.info('Horizontal step, from {} to {}.'.format(steps[index - 1], coordinate))
                        new_board.put_piece(steps[index - 1], 0)
                        new_board.put_piece(coordinate, piece_type)
                    else:
                        logger.error('Illegal move! From {} to {}.'.format(steps[index - 1], coordinate))
                        return new_board
                # Jump over other piece
                else:
                    mid_coordinate = (int((coordinate[0] + steps[index - 1][0]) / 2),
                                      int((coordinate[1] + steps[index - 1][1]) / 2))
                    mid_type = new_board.board[mid_coordinate[0]][mid_coordinate[1]]
                    if mid_type == 0:
                        logger.error('Can not jump over {}.'.format(mid_coordinate))
                        return new_board
                    if row_diff == 2 and col_diff == 0:
                        logger.info('Vertical jump, from {} to {}.'.format(steps[index - 1], coordinate))
                        new_board.put_piece(steps[index - 1], 0)
                        new_board.put_piece(coordinate, piece_type)
                        if mid_type != piece_type:
                            new_board.put_piece(mid_coordinate, 0)
                            logger.info('Capture piece at {}.'.format(mid_coordinate))
                    elif row_diff == 0 and col_diff == 2:
                        logger.info('Horizontal jump, from {} to {}.'.format(steps[index - 1], coordinate))
                        new_board.put_piece(steps[index - 1], 0)
                        new_board.put_piece(coordinate, piece_type)
                        if mid_type != piece_type:
                            new_board.put_piece(mid_coordinate, 0)
                            logger.info('Capture piece at {}.'.format(mid_coordinate))
                    else:
                        logger.error('Illegal move! From {} to {}.'.format(steps[index - 1], coordinate))
                        return new_board
        return new_board

    def is_terminated(self) -> bool:
        """
        See if this ChessBoard is GameOver
        TODO: Detect victory before turn 200.
        :return: bool
        """
        if self.game_turn < 200:
            return False
        remain_black = len(self.get_pieces(get_my_piece=True))
        remain_white = len(self.get_pieces(get_my_piece=False))
        if remain_black == 0 or remain_white == 0:
            print('Terminate before turn 200!')
            return True

        return True

    def getPossibleActions(self):
        """
        MCTS library function
        :return:
        """
        possible_actions = []
        for x in self.generate_outcomes():
            possible_actions.append(tuple(x))
        return possible_actions

    def takeAction(self, action):
        """
        MCTS library function
        :param action: Action() is hashable list of tuple
        :return:
        """
        new_state = self.move_piece(list(action))
        return new_state

    def isTerminal(self):
        """
        MCTS library function
        :return: bool
        """
        my_pieces = self.get_pieces(get_my_piece=True)
        op_pieces = self.get_pieces(get_my_piece=False)

        if self.is_black:
            if all([x[1] > 5 for x in my_pieces]) and len(my_pieces) > 0:
                return True
            if all([x[1] < 2 for x in op_pieces]) and len(op_pieces) > 0:
                return True
        else:
            if all([x[1] < 2 for x in my_pieces]) and len(my_pieces) > 0:
                return True
            if all([x[1] > 5 for x in op_pieces]) and len(op_pieces) > 0:
                return True
        if self.game_turn > self.mcts_turn + 4:
            return True
        elif len(op_pieces) == 0:
            return False
        elif len(my_pieces) == 0:
            return False
        else:
            return False

    def getReward(self):
        """
        MCTS library function
        :return: evaluation score
        """
        return self.evaluate

    @property
    def evaluate(self) -> int:
        """
        @property means you can just call ChessBoard.evaluate , without()
        evaluation score: larger means the winning chance is bigger
        TODO: Evaluation Function.
        :return: score of this board(state)
        """
        if self.is_black is True:
            my_color = 1  # Playing as black
            op_color = 2  # Opponent as white
        else:
            my_color = 2  # Playing as white
            op_color = 1  # Opponent as white

        # Get all pieces location
        my_pieces = self.get_pieces(get_my_piece=True)
        op_pieces = self.get_pieces(get_my_piece=False)
        for row_idx, row in enumerate(self.board):
            for col_idx, piece in enumerate(row):
                if piece == my_color:
                    my_pieces.append((row_idx, col_idx))
                elif piece == op_color:
                    op_pieces.append((row_idx, col_idx))

        final_score = 0
        """
        存活：我方加分，對手扣分
                每個存活棋子：s1
                    跨越中線：+s2
                        抵達禁區：+s3
                            禁區角落（無敵）：+s4
        """
        s1 = 200
        s2 = 50
        s3 = 50
        s4 = 300
        if self.is_black is True:
            for p in my_pieces:
                final_score += s1
                if p[1] > 3:
                    final_score += s2
                    if p[1] > 5:
                        final_score += s3
                        if p == (0, 7) or p == (7, 7):
                            final_score += s4
            for p in op_pieces:
                final_score -= s1
                if p[1] < 4:
                    final_score -= s2
                    if p[1] < 2:
                        final_score -= s3
                        if p == (0, 0) or p == (7, 0):
                            final_score -= s4
        else:  # is white
            for p in my_pieces:
                final_score += s1
                if p[1] < 4:
                    final_score += s2
                    if p[1] < 2:
                        final_score += s3
                        if p == (0, 0) or p == (7, 0):
                            final_score += s4
            for p in op_pieces:
                final_score -= s1
                if p[1] > 3:
                    final_score -= s2
                    if p[1] > 5:
                        final_score -= s3
                        if p == (0, 7) or p == (7, 7):
                            final_score -= s4

        """
        防禦:
        此處不計算對手
            2x2堡壘方陣：+d1
                回合50之後，轉進攻：
                    Approaching score:方陣越靠近禁區越高分：+d2 * col_idx
                    前進時，轉閃電陣不扣分：+d1
        """
        d1 = 100
        d2 = 200
        # xo
        # oo
        for p in my_pieces:
            if (p[0] + 1, p[1]) in my_pieces:
                if (p[0], p[1] + 1) in my_pieces:
                    if (p[0] + 1, p[1] + 1) in my_pieces:
                        logger.debug('Detect a 2x2 fortify.')
                        final_score += d1
                        # Approaching score
                        if self.game_turn > 20:
                            mid_idx = p[0] + 0.5
                            if self.is_black:
                                final_score += mid_idx * d2
                            else:
                                final_score += (7 - mid_idx) * d2

        # after turn 50
        if self.game_turn > 20:
            # xo
            #  oo
            for p in my_pieces:
                if (p[0], p[1] + 1) in my_pieces:
                    if (p[0], p[1] + 2) in my_pieces:
                        if (p[0] + 1, p[1] + 1) in my_pieces:
                            final_score += d1
                            # Approaching score
                            mid_idx = p[0] + 1
                            if self.is_black:
                                final_score += mid_idx * d2
                            else:
                                final_score += (7 - mid_idx) * d2
            #  oo
            # xo
            for p in my_pieces:
                if (p[0], p[1] + 1) in my_pieces:
                    if (p[0] - 1, p[1] + 1) in my_pieces:
                        if (p[0] - 1, p[1] + 2) in my_pieces:
                            final_score += d1
                            # Approaching score
                            mid_idx = p[0] + 1
                            if self.is_black:
                                final_score += mid_idx * d2
                            else:
                                final_score += (7 - mid_idx) * d2
        """
        進攻:
        此處不計算對手
            回合150之後，進攻分數加倍：
                每個過中線：+150
                每個禁區：+200
            蛇形滲透陣：
                2x1緊貼上下邊（無敵）：+400
                    每靠近禁區一格：+50
        """
        if self.game_turn > 0:
            for p in my_pieces:
                if self.is_black is True:
                    if p[1] > 1:
                        final_score += p[1] * 50
                    if p[1] > 5:
                        final_score += 200
                else:
                    if p[1] < 6:
                        final_score += (7 - p[1]) * 50
                    if p[1] < 2:
                        final_score += 200

        top_snake = False
        bot_snake = False
        for p in my_pieces:
            if p[0] == 0 and not top_snake:
                top_snake = True
                if (p[0], p[1] + 1) in my_pieces:
                    final_score += 500
                    if self.is_black:
                        final_score += p[1] * 100
                    else:
                        final_score += (7 - p[1]) * 100
            if p[0] == 7 and not bot_snake:
                bot_snake = True
                if (p[0], p[1] + 1) in my_pieces:
                    final_score += 500
                    if self.is_black:
                        final_score += p[1] * 100
                    else:
                        final_score += (7 - p[1]) * 100

        return final_score

    def get_pieces(self, get_my_piece: bool = True):
        """
        Get all pieces according to self.is_black.
        Can also get opponent's pieces.
        :param get_my_piece: Return my pieces or opponent's pieces.
        :return: [Piece Location tuple]
        """
        pieces = list()

        if get_my_piece != self.is_black:
            color = 2
        else:
            color = 1

        for row_idx, row in enumerate(self.board):
            for col_idx, piece in enumerate(row):
                if piece == color:
                    pieces.append((row_idx, col_idx))

        return pieces

    @staticmethod
    def get_next_hop(steps: [tuple], target: int, dst: list, previous_board: 'ChessBoard'):
        """
        Get next hop and middle piece coordinate & type.
        :param steps: current all steps, list[tuples]
        :param target: 0 for vertical hop, 1 for horizontal hop, int
        :param dst: vertical and horizontal distance from current location to destination(+2 or -2 or 0), list
        :param previous_board: previous_board: chess board after previous hop, ChessBoard
        :return: next hop: tuple, mid_coordinate: tuple, mid_type: int
        """
        last_step = len(steps) - 1
        row_mid = int(dst[0] / 2)
        col_mid = int(dst[1] / 2)
        mid_coordinate = None
        mid_type = None
        next_hop = None
        # Check range
        if -1 < steps[last_step][target] + dst[target] < 8:
            # Check destination
            if previous_board.board[steps[last_step][0] + dst[0]][steps[last_step][1] + dst[1]] == 0:
                # Check middle
                if previous_board.board[steps[last_step][0] + row_mid][steps[last_step][1] + col_mid] != 0:
                    mid_coordinate = (steps[last_step][0] + row_mid, steps[last_step][1] + col_mid)
                    mid_type = previous_board.board[steps[last_step][0] + row_mid][steps[last_step][1] + col_mid]
                    if len(steps) < 2:
                        next_hop = (steps[last_step][0] + dst[0], steps[last_step][1] + dst[1])
                    elif steps[last_step][target] + dst[target] != steps[last_step - 1][target]:
                        # Do not jump back.
                        next_hop = (steps[last_step][0] + dst[0], steps[last_step][1] + dst[1])

        return next_hop, mid_coordinate, mid_type

    def get_hop_recursively(self, steps: [tuple], hops: int, previous_board: 'ChessBoard'):
        """
        Get all possible hops recursively.
        :param steps: current all steps, list[tuples]
        :param hops: current number of hops, int
        :param previous_board: chess board after previous hop, ChessBoard
        :return: [Step tuple]
        """
        hops += 1
        if hops >= 99 or previous_board.game_turn >= 200:
            yield steps

        last_step = len(steps) - 1
        next_hop = list()  # Get all possible next hops.

        piece_type = previous_board.board[steps[last_step][0]][steps[last_step][1]]
        mid_coordinates = list()
        mid_types = list()

        # Hop above
        target_coordinate = 0  # vertical hop
        hop, mid_coordinate, mid_type = self.get_next_hop(steps, target_coordinate, [-2, 0], previous_board)
        if hop is not None:
            next_hop.append(hop)
            mid_coordinates.append(mid_coordinate)
            mid_types.append(mid_type)

        # Hop below
        hop, mid_coordinate, mid_type = self.get_next_hop(steps, target_coordinate, [2, 0], previous_board)
        if hop is not None:
            next_hop.append(hop)
            mid_coordinates.append(mid_coordinate)
            mid_types.append(mid_type)

        # Hop left
        target_coordinate = 1  # horizontal hop
        hop, mid_coordinate, mid_type = self.get_next_hop(steps, target_coordinate, [0, -2], previous_board)
        if hop is not None:
            next_hop.append(hop)
            mid_coordinates.append(mid_coordinate)
            mid_types.append(mid_type)

        # Hop right
        hop, mid_coordinate, mid_type = self.get_next_hop(steps, target_coordinate, [0, 2], previous_board)
        if hop is not None:
            next_hop.append(hop)
            mid_coordinates.append(mid_coordinate)
            mid_types.append(mid_type)

        if len(steps) > 1:
            yield steps

        for hop, mid_type, mid_coordinate in zip(next_hop, mid_types, mid_coordinates):
            next_steps = copy.deepcopy(steps)
            next_steps.append(hop)
            new_board = ChessBoard(input_board=previous_board.board, game_turn=previous_board.game_turn + 1)
            new_board.put_piece((steps[last_step][0], steps[last_step][1]), 0)
            new_board.put_piece(hop, piece_type)
            if mid_types != piece_type:
                new_board.put_piece(mid_coordinate, 0)
            yield from self.get_hop_recursively(next_steps, hops, new_board)

    def generate_outcomes(self):
        """
        Get all possible next moves.
        :return: [Step tuple]
        """

        pieces = self.get_pieces(get_my_piece=self.is_black == self.is_black_turn)

        if len(pieces) == 0:
            yield [(0, 0)]

        for piece in pieces:
            # Check adjacency
            if piece[0] - 1 > -1:  # Check piece above
                if self.board[piece[0] - 1][piece[1]] is 0:
                    yield [piece, (piece[0] - 1, piece[1])]
            if piece[0] + 1 < 8:  # Check piece below
                if self.board[piece[0] + 1][piece[1]] is 0:
                    yield [piece, (piece[0] + 1, piece[1])]
            if piece[1] - 1 > -1:  # Check left piece
                if self.board[piece[0]][piece[1] - 1] is 0:
                    yield [piece, (piece[0], piece[1] - 1)]
            if piece[1] + 1 < 8:  # Check right piece
                if self.board[piece[0]][piece[1] + 1] is 0:
                    yield [piece, (piece[0], piece[1] + 1)]

            # Check hop
            new_board = ChessBoard(input_board=self.board, game_turn=self.game_turn + 1)
            for hop in self.get_hop_recursively([piece], 0, new_board):
                yield hop

    @staticmethod
    def white_piece():
        """
        :return: white circle in UTF-8
        """
        return "\u25CF"

    @staticmethod
    def black_piece():
        """
        :return: black circle in UTF-8
        """
        return "\u25CB"

    @staticmethod
    def get_piece_string(piece: int) -> str:
        """
        :return: piece color in string
        """
        piece_dict = {0: 'None', 1: 'Black', 2: 'White'}
        return piece_dict[piece]


class Action:
    def __init__(self, steps):
        # list of tuple
        self.steps = steps

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.steps == other.steps

    def __hash__(self):
        return hash(str(self.steps))

    def __getitem__(self, item):
        return self.steps.__getitem__(item)


if __name__ == '__main__':
    """
    # A random game simulation
    board = ChessBoard(new_board=True, is_black=True, is_black_turn=True)
    for _ in range(200):
        steps = list(board.generate_outcomes())
        step = random.choice(steps)
        board = board.move_piece(step)
        board.print_board()
    """

    board = ChessBoard(new_board=True)
    moves = board.generate_outcomes()

    step = next(moves)
    jump_test = board.move_piece(step)
    jump_test.print_board()

    step = next(moves)
    board.move_piece(step).print_board()

    step = next(moves)
    board.move_piece(step).print_board()

    step = next(moves)
    board.move_piece(step).print_board()

    moves = jump_test.generate_outcomes()
    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()

    step = next(moves)
    jump_test.move_piece(step).print_board()
