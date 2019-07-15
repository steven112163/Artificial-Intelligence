from mcts import *
from ChessBoard import *


class GameAgent:
    def __init__(self, is_black: bool, state: [list], game_turn: int):
        self.state = ChessBoard(input_board=state,
                                is_black=is_black,
                                is_black_turn=is_black,
                                game_turn=game_turn,
                                mcts_turn=game_turn)
        if is_black:
            self.player_color = 'Black'
        else:
            self.player_color = 'White'

    def get_step(self):
        mcts_core = mcts(timeLimit=1000)
        best_action = mcts_core.search(initialState=self.state)

        logger.info('Turn {} Player {} move: {}'.format(self.state.game_turn, self.player_color, best_action))
        return best_action


if __name__ == '__main__':
    global_state = ChessBoard(new_board=True, game_turn=0,
                              is_black=True, is_black_turn=True,
                              input_board=None)
    global_state.clear_board()
    global_state.put_piece((1, 0), 1)
    global_state.put_piece((3, 0), 1)
    global_state.put_piece((4, 0), 1)
    global_state.put_piece((1, 4), 1)
    global_state.put_piece((2, 4), 2)
    global_state.print_board()
    for x in range(200):
        player = GameAgent(is_black=global_state.is_black_turn,
                           state=global_state.board,
                           game_turn=x)
        global_state = global_state.move_piece(player.get_step())
        global_state.print_board()
