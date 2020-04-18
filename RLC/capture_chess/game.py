from RLC.capture_chess.environment import Board, compose_move 
import numpy as np

def play_fixed_role(white_player, black_player, nr_games):
    white_wins = 0
    black_wins = 0
    remise = 0
    for i in range(nr_games):
        if (i % 10 == 0):
            print(f'game {i} of {nr_games}')
        env = play_game(white_player, black_player)
        result = env.determine_winner()
        if result > 0:
            white_wins += 1
        elif result < 0:
            black_wins += 1
        else:
            remise += 1
    print("white wins: " + str(white_wins))
    print("black wins: " + str(black_wins))
    print("remises: "+ str(remise))
    return white_wins, black_wins, remise

def play_game(w_agent, b_agent, max_steps_agent = 50):
    """
    Play a game of capture chess
    Args:
        w_agent: Agent
                Agent playing white
        b_agent: Agent
            Agent playing black
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
            episode_end, reward = w_agent.make_move(env, True)
        else:
            episode_end, reward = b_agent.make_move(env, False)

        if (turn_white):
            turncount_w += 1
        else:
            turncount_b += 1
            if turncount_b > max_steps_agent:
                # terminate, too many steps.
                episode_end = True
                
        # switch agent
        turn_white = not turn_white

    return env
  