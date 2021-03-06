import chess
import chess.svg
import chess.pgn
import numpy as np

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5

def compose_move(move_from, move_to):
    return chess.Move(move_from, move_to)


class Board(object):

    def __init__(self, FEN=None):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
        """
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()

    def init_action_space(self):
        """
        Initialize the action space
        Returns:

        """
        self.action_space = np.zeros(shape=(64, 64))

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
        if self.board.turn:
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1
    
    def state(self):
        """
        Get state of board
        """
        return self.layer_board
    
    def to_pgn(self):
        return chess.pgn.Game.from_board(self.board)
    
    def to_svg(self):
        return chess.svg.board(board = self.board, size=400)
    
    def to_file(self, file_loc):
        """
        Writes pgn of board to file for easy visualisation.
        """
        with open(file_loc, "w") as f:
            f.write(str(self.to_pgn()))
    
    def determine_winner(self):
        """
        positive is white wins, negative is black wins, zero is remise.
        """
        return self.get_material_value()


    def step(self, action):
        """
        Run a step from an Agent.
        Args:
            action: tuple of 2 integers
                Move from, Move to

        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: int
                Difference in material value after the move
        """
        piece_balance_before = self.get_material_value()
        self.board.push(action)
        self.init_layer_board()
        piece_balance_after = self.get_material_value()
        if self.board.result() == "*":
            capture_reward = piece_balance_after - piece_balance_before
            if self.board.result() == "*":
                reward = 0 + capture_reward
                episode_end = False
            else:
                reward = 0 + capture_reward
                episode_end = True
        else:
            capture_reward = piece_balance_after - piece_balance_before
            reward = 0 + capture_reward
            episode_end = True
        if self.board.is_game_over():
            reward = 0
            episode_end = True
        return episode_end, reward

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.

        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        # use chess legal moves generator to generate legal actions based on current state of the board.
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0], move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen
    
    def validate_move(self, move):
        """
        Checks if the move is valid, if not will return a random move.
        """
        
        moves = [x for x in self.board.generate_legal_moves() if \
                    x.from_square == move.from_square and x.to_square == move.to_square]
        if len(moves) == 0:  # If all legal moves have negative action value, explore.
            move = self.get_random_action()
        else:
            move = np.random.choice(moves)  # If there are multiple valid moves, pick a random one.
        return move

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()
