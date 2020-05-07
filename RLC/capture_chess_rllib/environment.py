import chess
import chess.svg
import chess.pgn
import numpy as np

from ray.rllib.agents.pg.pg import PGTrainer
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Dict

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


class CaptureChessEnv(MultiAgentEnv):
    """Two-player environment for capture chess."""
    observation_space = Dict({
        "action_mask": Box(np.float32(0), np.float32(1), (4096,), dtype=np.float32),
        # 8 layers for different types and additional info, each has position on 8 x 8 board.
        "real_obs": Box(np.float32(-1.0), np.float32(1.0), (8, 8, 8), dtype=np.float32),
    })
    # move_from and move_to action each is position on board.
    action_space = Box(0, 1, (4096,), dtype=np.bool)

    def __init__(self, FEN=None):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
        """
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()

        self.player1 = "player1"
        self.player2 = "player2"

        self.reset()

    def init_action_space(self):
        """
        Initialize the action space
        Returns:

        """
        self.action_space = np.zeros(shape=(4096,))

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
            # Encode move as Nr between 0 and 1.
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
        return chess.svg.board(board=self.board, size=400)

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

    def reset(self):
        """
        Reset the environment
        Returns: observation

        """
        self.num_moves = 0
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()
        self.prev_reward = 0
        obs = {
            # player 1 starts.
            self.player1: {"real_obs": self.state(), "action_mask": self.project_legal_moves()},
        }
        return obs

    def _make_move(self, move_space):
        possible_actions = np.where(self.action_space == 1)
        chosen_action = np.random.choice(np.flatnonzero(
            move_space[possible_actions] == move_space[possible_actions].max()))
        move_idx = possible_actions[0][chosen_action]
        move = compose_move(move_idx % 64, move_idx // 64)

        piece_balance_before = self.get_material_value()
        self.board.push(move)
        self.init_layer_board()
        piece_balance_after = self.get_material_value()
        if self.board.result() == "*":
            capture_reward = piece_balance_after - piece_balance_before
            if self.board.result() == "*":
                episode_end = False
            else:
                episode_end = True
        else:
            capture_reward = piece_balance_after - piece_balance_before
            episode_end = True
        if self.board.is_game_over():
            episode_end = True
        return episode_end, capture_reward

    def step(self, action_dict):
        """
        Run a step.
        """
        def check_end(episode_end, num_moves):
            return episode_end or num_moves >= 100 # 50 steps each player.

        # Already increase num moves for correct check_end.
        self.num_moves += 1

        if action_dict.get(self.player1) is not None:
            episode_end, reward1 = self._make_move(action_dict[self.player1])
            obs = {self.player2: {"real_obs": self.state(),
                                  "action_mask": self.project_legal_moves()}}
            rewards = {
                # inverse reward for second player.
                self.player2: - (reward1 + self.prev_reward),
            }
            if check_end(episode_end, self.num_moves):
                obs[self.player1] = {"real_obs": self.state(),
                                     "action_mask": self.project_legal_moves()}
                rewards[self.player1] = reward1

            self.prev_reward = reward1
        elif action_dict.get(self.player2) is not None:
            episode_end, reward2 = self._make_move(action_dict[self.player2])
            obs = {self.player1: {"real_obs": self.state(),
                                  "action_mask": self.project_legal_moves()}}
            rewards = {
                self.player1: reward2 + self.prev_reward,
            }
            if check_end(episode_end, self.num_moves):
                obs[self.player2] = {"real_obs": self.state(),
                                     "action_mask": self.project_legal_moves()}
                # inverse reward for second player.
                rewards[self.player2] = - reward2

            self.prev_reward = reward2
        print(rewards)
        done = {
            "__all__": check_end(episode_end, self.num_moves)
        }

        return obs, rewards, done, {}

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
        Returns: np.ndarray with shape (4096,)
        """
        self.action_space = np.zeros(shape=(4096,))
        # use chess legal moves generator to generate legal actions based on current state of the board.
        moves = [[x.from_square, x.to_square]
                 for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0] + 64 * move[1]] = 1
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

        moves = [x for x in self.board.generate_legal_moves() if
                 x.from_square == move.from_square and x.to_square == move.to_square]
        # If all legal moves have negative action value, explore.
        if len(moves) == 0:
            move = self.get_random_action()
        else:
            # If there are multiple valid moves, pick a random one.
            move = np.random.choice(moves)
        return move
    
    def render(self, mode='human'):
        if mode == "human":
            return self.to_svg()
        else:
            raise NotImplementedError()
