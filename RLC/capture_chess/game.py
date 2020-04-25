from RLC.capture_chess.environment import Board, compose_move 
import numpy as np

def play_alternate_role(player1, player2, nr_games_each_role, print_results=True):
    player1_wins = 0
    player2_wins = 0
    remise = 0
    for i in range(nr_games_each_role):
        if (i % 10 == 0 and print_results):
            print(f'game {i} of {nr_games_each_role}')
        env = play_game(player1, player2)
        result = env.determine_winner()
        if result > 0:
            player1_wins += 1
        elif result < 0:
            player2_wins += 1
        else:
            remise += 1
        env = play_game(player2, player1)
        result = env.determine_winner()
        if result > 0:
            player2_wins += 1
        elif result < 0:
            player1_wins += 1
        else:
            remise += 1
    if print_results:
        print("player 1 wins: " + str(player1_wins))
        print("player 2 wins: " + str(player2_wins))
        print("remises: "+ str(remise))
    return player1_wins, player2_wins, remise
    
def play_fixed_role(white_player, black_player, nr_games, print_results=True):
    white_wins = 0
    black_wins = 0
    remise = 0
    for i in range(nr_games):
        if (i % 10 == 0 and print_results):
            print(f'game {i} of {nr_games}')
        env = play_game(white_player, black_player)
        result = env.determine_winner()
        if result > 0:
            white_wins += 1
        elif result < 0:
            black_wins += 1
        else:
            remise += 1
    if print_results:
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
    prev_state = 0
    reward_other = 0
    move_other = None
    while not episode_end:
        state_before_step = env.state()
        if (turn_white):
            move = w_agent.determine_move(env, True)
        else:
            move = b_agent.determine_move(env, False)
        # make move
        episode_end, reward = env.step(move)

        state_after_step = env.state()
        if (turn_white):
            turncount_w += 1

            # update black with info of white and black
            if (turncount_b != 0):
                # *_other is move of b_agent
                # invert rewards (for agent good means positive reward.)
                b_agent.update(prev_state, state_before_step, move_other, - reward_other, state_after_step, move, - reward)
        else:
            turncount_b += 1
            if turncount_b > max_steps_agent:
                # terminate, too many steps.
                episode_end = True
                
            # update white with info of white and black
            if (turncount_w != 0):
                # *_other is move of w_agent
                w_agent.update(prev_state, state_before_step, move_other, reward_other, state_after_step, move, reward)
        
        # switch agent
        turn_white = not turn_white
        prev_state = state_before_step
        reward_other = reward
        move_other = move

    w_agent.update_after_game()
    b_agent.update_after_game()
    return env
  