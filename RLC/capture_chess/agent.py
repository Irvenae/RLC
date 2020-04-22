from keras.models import Model, clone_model
from keras.layers import Input, Conv2D, Dense, Reshape, Dot, Activation, Multiply
from keras.optimizers import SGD
import numpy as np
import keras.backend as K

# import own modules
from RLC.capture_chess.environment import compose_move

# start hotfix module 'tensorflow._api.v2.config' has no attribute 'experimental_list_devices'
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
# end hotfix module 'tensorflow._api.v2.config' has no attribute 'experimental_list_devices' 



def policy_gradient_loss(cumulative_rewards):
    """
    Policy gradient loss is defined as the sum, over timesteps, of action
    log-probability (same as categorical_crossentropy) times the cumulative rewards from that timestep onward.
    """
    def modified_crossentropy(action, action_probs):
        cost = (K.categorical_crossentropy(action, action_probs, from_logits=False, axis=1) * cumulative_rewards)
        return K.mean(cost) # TODO(Irven): try with sum.
    return modified_crossentropy


#################################
#       Networks                #
#################################

def init_linear_network(lr):
    """
    Initialize a linear neural network
    Returns: model

    """
    optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    input_layer = Input(shape=(8, 8, 8), name='board_layer')
    reshape_input = Reshape((512,))(input_layer)
    output_layer = Dense(4096)(reshape_input)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def init_conv_network(lr):
    """
    Initialize a convolutional neural network
    Returns: model

    """
    optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    input_layer = Input(shape=(8, 8, 8), name='board_layer')
    inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
    inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
    flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
    flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
    output_dot_layer = Dot(axes=1)([flat_1, flat_2])
    output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def init_conv_pg(lr):
    """
    Convnet net for policy gradients
    Returns: model

    """
    optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    input_layer = Input(shape=(8, 8, 8), name='board_layer')
    cumulative_rewards = Input(shape=(1,), name='cumulative_rewards')
    legal_moves = Input(shape=(4096,), name='legal_move_mask')
    inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
    inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
    flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
    flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
    output_dot_layer = Dot(axes=1)([flat_1, flat_2])
    output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
    softmax_layer = Activation('softmax')(output_layer)
    legal_softmax_layer = Multiply()([legal_moves, softmax_layer])  # Select legal moves
    model = Model(inputs=[input_layer, cumulative_rewards, legal_moves], outputs=[legal_softmax_layer])
    model.compile(optimizer=optimizer, loss=policy_gradient_loss(cumulative_rewards))
    return model


    #################################
    #       Agents                  #
    #################################
class Agent(object):
    """
    Makes the decision of what action to take.
    """
    def __init__(self):
        pass
    
    def reset_for_game(self):
        """
        Needs to be called before each game.
        """
        raise NotImplementedError()

    def set_learn(self, learn):
        """
        Set learning on / off
        """
        raise NotImplementedError()

    def determine_move(self, env, white_player):
        """
        Determine next move
        Args:
            env: Board
                environment of board.
            white_player: boolean
                Is the current player white?
        Returns: move
        """
        raise NotImplementedError()

    def update(self, prev_state, state, move, reward, state_other, move_other, reward_other, minibatch_size=128):
        """
        Update the agent (learning network) using experience replay. Set the sampling probs with the td error.
        Args:
           prev_state: 
                previous state board
            state:
                state of board after move
            move:
                performed move (move.from_square, move.to_square)
            reward:
                reward of the move
           state_other:
                state of board after others move 
            move_other:
                performed move other agent(move.from_square, move.to_square)
            reward_other:
                reward of the other agent move
        """
        raise NotImplementedError()
    
    def update_after_game(self):
        """
        Called after game.
        """
        raise NotImplementedError()

class RandomAgent(Agent):
    """
    Plays a random move
    """

    def reset_for_game(self):
        """
        Needs to be called before each game.
        """
        pass

    def set_learn(self, learn):
        """
        Set learning on / off
        """
        pass

    def determine_move(self, env, white_player):
        """
        Determine next move
        Args:
            env: Board
                environment of board.
            white_player: boolean
                Is the current player white?
        Returns: move
        """
        return env.get_random_action()
    
    def update(self, prev_state, state, move, reward, state_other, move_other, reward_other, minibatch_size=128):
        """
        Update the agent (learning network) using experience replay. Set the sampling probs with the td error.
        Args:
           prev_state: 
                previous state board
            state:
                state of board after move
            move:
                performed move (move.from_square, move.to_square)
            reward:
                reward of the move
           state_other:
                state of board after others move 
            move_other:
                performed move other agent(move.from_square, move.to_square)
            reward_other:
                reward of the other agent move
        """
        pass

    def update_after_game(self):
        """
        Called after game.
        """
        pass


class QExperienceReplayAgent(Agent):
    """
    Apply Q learning using a feeding and learning network with experience replay. 
    With subsequent games a memory is kept. This memory is then used to sample.
    In each step we update the learning network using the samples and after a
    number of steps we update the feeding network with the learning network.

    The policy implied by Q-Learning is deterministic. This means that Q-Learning can’t learn stochastic policies (which is how we see the random steps).
    """
    
    def __init__(self, network = "conv", c_feeding = 20, memsize = 1000, gamma = 0.5, lr = 0.01, verbose = 0, log_dir = None):
        """
        Args:
            network: string
                Network to use
            c_feeding: int
                update feeding network after c_feeding updates.
            memsize: int
                total states to hold in memory for experience replay.
            gamma: float
                Temporal discount factor
            lr: float
                Learning rate, ideally around 0.1
            verbose: int
                verbose output: 0 or 1.
            log_dir: 
                directory for logging
        """
        super().__init__()
        self.network = network
        self.c_feeding = c_feeding
        self.memsize = memsize
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.writer = None
        if (log_dir):
            self.writer = tf.summary.create_file_writer(log_dir)
            self.writer_step = 0
            self.writer_step_start_episode = self.writer_step 
        
        self.feeding_count = 0
        self.game_count = 0 # only learning games are counted.
        self.memory = []
        self.sampling_probs = []
        self.reward_trace = []
        
        self.set_learn(True)
        self.init_network() # set learning model
        self.fix_model() # set feeding model
        self.set_default_epsilon_function()

    #################################
    #       External functions      #
    #################################

    def set_default_epsilon_function(self):
        self.set_epsilon_function(self.default_epsilon_func)

    def set_epsilon_function(self, func):
        """
        Args:
            func: Function
                function with game_count as input. (see default_epsilon_func in __init__)
        """
        self.epsilon_func = func
    
    def get_epsilon(self):
        """
        Get epsilon of current game.
        """
        return self.epsilon_func(self.game_count)
    
    def set_learn(self, learn):
        """
        Set learning on / off
        """
        self.learn = learn

    def reset_for_game(self):
        """
        Needs to be called before each game.
        Each reset will be seen as a new game start.
        The feeding network will be updated if necessary.
        """
        self.reward_trace = []
        if self.learn:
            self.game_count += 1
            self.feeding_count += 1
        if self.feeding_count % self.c_feeding == 0 and self.feeding_count != 0:
            print("updated feeding network")
            self.fix_model()
            self.feeding_count = 0
    
    def determine_move(self, env, white_player):
        """
        Determine next move
        Args:
            env: Board
                environment of board.
            white_player: boolean
                Is the current player white?
        Returns: move
        """
        eps = self.get_epsilon()
        explore = np.random.uniform(0, 1) < eps  # determine whether to explore
        state = env.state()
        if np.array_equal(state, 0):
            raise Exception("Game has already ended!")
        if explore:
            move = env.get_random_action()
            move_from = move.from_square
            move_to = move.to_square
        else:
            action_space = env.project_legal_moves()  # The environment determines which moves are legal

            action_values = self.get_action_values(np.expand_dims(state, axis=0))
            action_values = np.reshape(np.squeeze(action_values), (64, 64))
            action_values = np.multiply(action_values, action_space)
            # get position with maximal / minimal index from 64 x 64 matrix.
            # Store row index in move_from and column index in move_to.
            if white_player:
                move_from = np.argmax(action_values, axis=None) // 64
                move_to = np.argmax(action_values, axis=None) % 64
            else:
               move_from = np.argmin(action_values, axis=None) // 64
               move_to = np.argmin(action_values, axis=None) % 64 

        return env.validate_move(compose_move(move_from, move_to))

    def update(self, prev_state, state, move, reward, state_other, move_other, reward_other, minibatch_size=256):
        """
        Update the agent (learning network) using experience replay. Set the sampling probs with the td error.
        Args:
           prev_state: 
                previous state board
            state:
                state of board after move
            move:
                performed move (move.from_square, move.to_square)
            reward:
                reward of the move
           state_other:
                state of board after others move 
            move_other:
                performed move other agent(move.from_square, move.to_square)
            reward_other:
                reward of the other agent move
        """
        if self.learn:
            self.reward_trace.append(reward + reward_other)
            # update memory for experienced replay 
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)
            self.memory.append([prev_state, (move.from_square, move.to_square), reward + reward_other, state_other])
            self.sampling_probs.append(1)

            # update model with prioritized experienced replay
            minibatch, indices = self.sample_memory(minibatch_size)
            td_errors = self.update_model(minibatch)
            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])
            
            # save abs sum of errors in graph:
            if (self.writer):
                with self.writer.as_default():
                    tf.summary.scalar('mean time difference error', data=np.mean(np.abs(td_errors)), step=self.writer_step)
                    self.writer.flush()
                    self.writer_step += 1

    def update_after_game(self):
        """
        Called after game.
        """
        if self.learn:
            # calculate cumulative reward
            discounts = np.array([ self.gamma ** i for i in range(0, len(self.reward_trace))])
            cumulative_rewards = [ np.sum(discounts[:(len(self.reward_trace) - i)] * self.reward_trace[i:]) for i in range(0, len(self.reward_trace))]
            if (self.writer):
                with self.writer.as_default():
                    for i, cumulative_reward in enumerate(cumulative_rewards):
                        tf.summary.scalar('cumulative reward', cumulative_reward, step=self.writer_step_start_episode + i )
                    tf.summary.scalar('mean reward episode', np.mean(self.reward_trace), step=self.game_count)
                    tf.summary.scalar('length episode', len(self.reward_trace), step=self.game_count)
                    self.writer.flush()

                self.writer_step_start_episode += len(cumulative_rewards)
    

    #################################
    #       Internal functions      #
    #################################
    
    def init_network(self):
        """
        Initialize the network
        Returns:

        """
        if self.network == 'linear':
            self.model = init_linear_network(self.lr)
        elif self.network == 'conv':
            self.model = init_conv_network(self.lr)
        else:
            raise NotImplementedError()
    
    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """
        optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
        self.feeding_model = clone_model(self.model)
        self.feeding_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.feeding_model.set_weights(self.model.get_weights())

    # determine default epsilon function
    @staticmethod
    def default_epsilon_func(k):
        c = 0.5 
        min_epsilon = 0.05
        if k == 0:
            epsilon = 1
        else:
            epsilon = max(c / k, min_epsilon)
        return epsilon

    def get_action_values(self, state):
        """
        Get action values of a state
        Args:
            state: np.ndarray with shape (8,8,8)
                layer_board representation

        Returns:
            action values

        """
        return self.feeding_model.predict(state) + np.random.randn() * 1e-9

    def sample_memory(self, minibatch_size):
        """
        Get a sample from memory for experience replay
        Args:
            minibatch_size: int
                size of the minibatch

        Returns: tuple
            a mini-batch of experiences (list)
            indices of chosen experiences

        """
        sample_probs = [self.sampling_probs[n] / np.sum(self.sampling_probs) for n in range(len(self.sampling_probs))]
        n_values = min(minibatch_size, len(self.memory))
        indices = np.random.choice(range(len(self.memory)), n_values, replace=True, p=sample_probs)

        minibatch = []
        for i in indices:
            minibatch.append(self.memory[i])
        return minibatch, indices

    def update_model(self, minibatch):
        """
        Update the Q-network using samples from the minibatch. Does not take into account the other 
        players move.
        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards and new states.

        Returns:
            td_errors: np.array
                array of temporal difference errors

        """

        # Prepare separate lists
        states, moves, rewards, new_states = [], [], [], []
        td_errors = []
        episode_did_not_ends = []
        for sample in minibatch:
            states.append(sample[0])
            moves.append(sample[1])
            rewards.append(sample[2])
            new_states.append(sample[3])

            # Episode end detection
            if np.array_equal(sample[3], 0):
                episode_did_not_ends.append(0)
            else:
                episode_did_not_ends.append(1)

        # The Q target
        q_target = np.array(rewards) + np.array(episode_did_not_ends) * self.gamma * np.max(
            self.feeding_model.predict(np.stack(new_states, axis=0)), axis=1)

        # The Q value for the remaining actions
        q_state = self.model.predict(np.stack(states, axis=0))  # batch x 64 x 64

        # Combine the Q target with the other Q values.
        q_state = np.reshape(q_state, (len(minibatch), 64, 64))
        for idx, move in enumerate(moves):
            td_errors.append(q_state[idx, move[0], move[1]] - q_target[idx])
            q_state[idx, move[0], move[1]] = q_target[idx]
        q_state = np.reshape(q_state, (len(minibatch), 4096))

        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=np.stack(states, axis=0), y=q_state, epochs=1, verbose=self.verbose)

        return td_errors


class ReinforceAgent(Agent):
    """
    REINFORCE: REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility
    REINFORCE is the fundamental policy gradient algorithm.
    """
    def __init__(self, memsize = 1000, gamma = 0.5, lr = 0.3, verbose = 0, log_dir = None):
        """
        Args:
            memsize: int
                total states to hold in memory for experience replay.
            gamma: float
                Temporal discount factor
            lr: float
                Learning rate, ideally around 0.1
            verbose: int
                verbose output: 0 or 1.
            log_dir: 
                directory for logging
        """
        super().__init__()
        self.memsize = memsize
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.writer = None
        if (log_dir):
            self.writer = tf.summary.create_file_writer(log_dir)
            self.writer_step = 0
            self.writer_step_start_episode = 0
        
        self.feeding_count = 0
        self.game_count = 0 # only learning games are counted.
        self.memory_mean_rewards = []
        self.episode_data = []
        
        self.set_learn(True)
        self.model = init_conv_pg(self.lr)
    
    class EpisodeData():
        def __init__(self, state, move, reward, action_space):
            self.state = state
            self.move = move
            self.reward = reward
            self.action_space = action_space

    #################################
    #       External functions      #
    #################################

    def set_learn(self, learn):
        """
        Set learning on / off
        """
        self.learn = learn

    def reset_for_game(self):
        """
        Needs to be called before each game.
        Each reset will be seen as a new game start.
        The feeding network will be updated if necessary.
        """
        self.episode_data = []
        if self.learn:
            self.game_count += 1
    
    def determine_move(self, env, white_player):
        """
        Determine next move
        Args:
            env: Board
                environment of board.
            white_player: boolean
                Is the current player white?
        Returns: move
        """
        state = env.state()
        if np.array_equal(state, 0):
            raise Exception("Game has already ended!")
       
        action_space = env.project_legal_moves()  # The environment determines which moves are legal
        self.temp_action_space = action_space # hack to get action space in update function.
        action_probs = self.model.predict([np.expand_dims(state, axis=0),
                                                     np.zeros((1, 1)),
                                                     action_space.reshape(1, 4096)])
        action_probs = action_probs / action_probs.sum()
        # if not white_player:

        # get position from 64 x 64 matrix.
        # Store row index in move_from and column index in move_to.
        move = np.random.choice(range(4096), p=np.squeeze(action_probs))
        move_from = move // 64
        move_to = move % 64

        if (self.learn and self.writer):
            with self.writer.as_default():
                tf.summary.scalar('probability max move', np.max(action_probs), step=self.writer_step )
            self.writer_step += 1

        return env.validate_move(compose_move(move_from, move_to))

    def update(self, prev_state, state, move, reward, state_other, move_other, reward_other, minibatch_size=256):
        """
        Update the agent (learning network) using experience replay. Set the sampling probs with the td error.
        Args:
           prev_state: 
                previous state board
            state:
                state of board after move
            move:
                performed move (move.from_square, move.to_square)
            reward:
                reward of the move
           state_other:
                state of board after others move 
            move_other:
                performed move other agent(move.from_square, move.to_square)
            reward_other:
                reward of the other agent move
        """
        if self.learn:
            # add data that will be used at end of episode
            data = self.EpisodeData(prev_state, move, reward + reward_other, self.temp_action_space)
            self.episode_data.append(data)

    def update_after_game(self):
        """
        Called after game.
        """
        if self.learn:
            cumulative_rewards = self.update_model()
            
            if (self.writer):
                with self.writer.as_default():
                    for i, cumulative_reward in enumerate(cumulative_rewards):
                        tf.summary.scalar('cumulative reward', cumulative_reward, step=self.writer_step_start_episode + i )
                    tf.summary.scalar('mean reward episode', np.mean(cumulative_rewards), step=self.game_count)
                    tf.summary.scalar('length episode', len(cumulative_rewards), step=self.game_count)
                    self.writer.flush()

                self.writer_step_start_episode += len(cumulative_rewards)


    #################################
    #       Internal functions      #
    #################################

    def update_model(self):
        """
        Update model with Monte Carlo Policy Gradient algorithm needs data of entire episode.
        
        Returns:
            cumulative_rewards in episode
        """
         
        n_steps = len(self.episode_data)
        discounts = np.array([ self.gamma ** i for i in range(0, len(self.episode_data))])
        cumulative_rewards = [ np.sum(discounts[:(n_steps - i)] * self.episode_data[i].reward) for i in range(0, n_steps)]
        states = [ self.episode_data[i].state for i in range(0, n_steps)]
        action_spaces = [ self.episode_data[i].action_space.reshape(1, 4096) for i in range(0, n_steps)]
        targets = np.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = self.episode_data[t].move
            targets[t, action.from_square, action.to_square] = 1

        mean_cumulative_reward = np.mean(cumulative_rewards)
        if len(self.memory_mean_rewards) > self.memsize:
                self.memory_mean_rewards.pop(0)
        self.memory_mean_rewards.append(mean_cumulative_reward)

        train_cumulative_rewards = np.stack(cumulative_rewards, axis=0) - np.mean(self.memory_mean_rewards)
        # ideally this can be optimized by using the advantage function = q(s,a) - v(s)
                
        targets = targets.reshape((n_steps, 4096))
        self.model.fit(x=[np.stack(states, axis=0),
                          train_cumulative_rewards,
                          np.concatenate(action_spaces, axis=0)
                          ],
                       y=[np.stack(targets, axis=0)],
                       verbose=self.verbose
                       )
        return cumulative_rewards
