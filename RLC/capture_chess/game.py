from RLC.capture_chess.environment import Board, compose_move 
import numpy as np

def play_move(env, agent, white_player):
    """
    Play a single move of capture chess
    Args:
        env: Board
            environment
        state:
            Current state of the board.
        agent: Agent
            Agent that performs move and needs to be updated.
        white_player: boolean
                Is the current player white?

    Returns: board environment.
    """
    move = agent.next_move(env, white_player)
    # check if move is valid.
    moves = [x for x in env.board.generate_legal_moves() if \
                x.from_square == move.from_square and x.to_square == move.to_square]
    if len(moves) == 0:  # If all legal moves have negative action value, explore.
        move = env.get_random_action()
        move_from = move.from_square
        move_to = move.to_square
    else:
        move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
    return move

def play_game(w_agent, b_agent, learn = True, max_steps_agent = 50):
    """
    Play a game of capture chess
    Args:
        w_agent: Agent
                Agent playing white
        b_agent: Agent
            Agent playing black
        learn: boolean
            Let the agent learn
        max_steps_agent: int
            Maximum amount of steps per game for each agent.

    Returns: Board environment.
    """
    # reset game
    env = Board()
    w_agent.reset_for_game()
    b_agent.reset_for_game()

    # initialize loop values
    episode_end = False
    turncount_w = 0
    turncount_b = 0
    turn_white = True
    while not episode_end:
        state = env.state()
        if (turn_white):
            move = play_move(env, w_agent, True)
        else:
            move = play_move(env, b_agent, False)

        episode_end, reward = env.step(move)
        new_state = env.state()

        if (turn_white):
            turncount_w += 1
            if turncount_w > max_steps_agent:
                reward = 0
            if learn:
                # update with info of white and black
                w_agent.update_variables(state, new_state, move, reward)
                w_agent.update(turncount_w)
        else:
            turncount_b += 1
            if turncount_b > max_steps_agent:
                reward = 0
                # terminate, too many steps.
                episode_end = True
            if learn:
                # update with info of white and black
                b_agent.update_variables(state, new_state, move, reward)
                b_agent.update(turncount_b)
        
        # switch agent
        turn_white = not turn_white

    return env
  