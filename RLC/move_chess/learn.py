import numpy as np
import pprint

class Reinforce(object):

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_episode(self, state, max_steps=1e3, epsilon=0.1):
        """
        Play an episode of move chess
        :param state: tuple describing the starting state on 8x8 matrix
        :param max_steps: integer, maximum amount of steps before terminating the episode
        :param epsilon: exploration parameter
        :return: tuple of lists describing states, actions and rewards in a episode
        """
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False

        # Play out an episode
        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon)  # get the index of the next action
            action = self.agent.action_space[action_index]
            actions.append(action_index)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)

            #  avoid infinite loops
            if count_steps > max_steps:
                episode_end = True

        return states, actions, rewards

    def glie_eps_greedy(self, func, n_episodes=1000, c=0.5, min_epsilon=0.05):
        """
        Run the epsilon greedy behavior with GLIE property on func.
        You should send func with partial so only epsilon still needs to be added.
            from functools import partial
            glie_eps_greedy(partial(sarsa_td, gamma= 0.8, ...)) 
        :param n_episodes: int, amount of episodes to train
        :param c: constant to use in epsilon
        :return: finds the optimal policy for move chess
        """
        for k in range(1, n_episodes):
            epsilon = max(c / k, min_epsilon)
            func(epsilon=epsilon)

    def sarsa_td(self, epsilon=0.1, gamma=0.9, alpha=0.05, max_steps=1000):
        """
        Run the sarsa control algorithm (TD0), finding the optimal policy and action function
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param alpha: learning rate
        :param max_steps: max amount of steps in an episode
        :return: finds the optimal policy for move chess
        """
        state = (0, 0)
        self.env.state = state
        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1
            state = self.env.state
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            reward, episode_end = self.env.step(action)
            successor_state = self.env.state
            successor_action_index = self.agent.apply_policy(successor_state, epsilon)

            action_value = self.agent.action_function[state[0], state[1], action_index]
            successor_action_value = self.agent.action_function[successor_state[0],
                                                                successor_state[1], successor_action_index]
            delta = reward + gamma * successor_action_value - action_value

            self.agent.action_function[state[0], state[1], action_index] += alpha * delta
            self.agent.policy = self.agent.action_function.copy()
            if count_steps > max_steps:
                episode_end = True

    def sarsa_lambda(self, epsilon=0.1, gamma=0.9, alpha=0.05, lamb=0.8, max_steps=1000):
        """
        Run the sarsa control algorithm (TD lambda), finding the optimal policy and action function
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param alpha: learning rate
        :param lamb: lambda parameter describing the decay over n-step returns
        :param max_steps: max amount of steps in an episode
        """
        self.agent.E = np.zeros(shape=self.agent.action_function.shape)
        state = (0, 0)
        self.env.state = state
        action_index = self.agent.apply_policy(state, epsilon)
        action = self.agent.action_space[action_index]

        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1
            reward, episode_end = self.env.step(action)
            successor_state = self.env.state
            successor_action_index = self.agent.apply_policy(successor_state, epsilon)

            action_value = self.agent.action_function[state[0], state[1], action_index]
            if not episode_end:
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]
            else:
                successor_action_value = 0
            delta = reward + gamma * successor_action_value - action_value
            self.agent.E[state[0], state[1], action_index] += 1
            self.agent.action_function[state[0], state[1], action_index] += alpha * delta * self.agent.E[state[0], state[1], action_index]
            self.agent.E = gamma * lamb * self.agent.E

            state = successor_state
            action = self.agent.action_space[successor_action_index]
            action_index = successor_action_index
            self.agent.policy = self.agent.action_function.copy()
            if count_steps > max_steps:
                episode_end = True

    def q_learning(self, epsilon=0.1, gamma=0.9, alpha=0.05, max_steps=1000):
        """
        Run Q-learning (also known as sarsa-max, finding the optimal policy and value function
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param alpha: learning rate
        :param max_steps: max amount of steps in an episode
        :return: finds the optimal move chess policy
        """
        state = (0, 0)
        self.env.state = state
        action_index = self.agent.apply_policy(state, epsilon)
        action = self.agent.action_space[action_index]

        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1

            reward, episode_end = self.env.step(action)
            successor_state = self.env.state
            successor_action_index = np.argmax(self.agent.policy[successor_state[0], successor_state[1], :])

            action_value = self.agent.action_function[state[0], state[1], action_index]
            if not episode_end:
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]
            else:
                successor_action_value = 0

            delta = reward + gamma * successor_action_value - action_value
            self.agent.action_function[state[0], state[1], action_index] += alpha * delta

            state = successor_state
            action = self.agent.action_space[successor_action_index]
            action_index = successor_action_index
            self.agent.policy = self.agent.action_function.copy()
            if count_steps > max_steps:
                episode_end = True

    # TODO(Irven) does not work properly currently. Probably because multiple steps are equally good
    # the episode breaks down too guickly. We need to check why the episode breaks down so fast.
    def monte_carlo_importance_sampling(self, target_policy, epsilon=0.1, gamma = 0.9):
        state = (0, 0)
        self.env.state = state

        # Play out an episode
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for idx, state in reversed(list(enumerate(states))):
            if idx == len(states) -1:
                continue
            print(idx, state)
            action_index = actions[idx]

            G = gamma * G + rewards[idx]
            # Update weighted importance sampling formula denominator
            self.agent.N_action[state[0], state[1], action_index] += W
            
            delta = (G - self.agent.action_function[state[0], state[1], action_index])
            applied_weight = (W / self.agent.N_action[state[0], state[1], action_index])
            self.agent.action_function[state[0], state[1], action_index] += applied_weight * delta
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action_index != np.argmax(target_policy[state[0], state[1], :]):
                break
            W = W * 1./ (self.agent.policy[state[0], state[1], action_index] + 0.001)

    def monte_carlo_learning(self, epsilon=0.1, gamma = 0.9, first_visit=True):
        """
        Learn move chess through monte carlo control
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param first_visit: Boolean, count only from first occurence of state
        :return:
        """
        state = (0, 0)
        self.env.state = state

        # Play out an episode
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        discounts =  np.array([pow(gamma,i) for i in range(0, len(rewards))])
        visited_state_actions = set()
        # TODO(Irven) discounts calculation can be simplified by inversing order of loop.
        for idx, state in enumerate(states):
            action_index = actions[idx]
            if (state, action_index) not in visited_state_actions and first_visit:
                self.agent.N_action[state[0], state[1], action_index] += 1
                n = self.agent.N_action[state[0], state[1], action_index]
                forward_rewards = np.sum(discounts[:(len(rewards) - idx)] * rewards[idx:])
                expected_rewards = self.agent.action_function[state[0], state[1], action_index]
                delta = forward_rewards - expected_rewards
                self.agent.action_function[state[0], state[1], action_index] += ((1 / n) * delta)
                visited_state_actions.add(state)
            elif not first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(discounts[:(len(rewards) - idx)] * rewards[idx:])
                expected_rewards = self.agent.action_function[state[0], state[1], action_index]
                delta = forward_rewards - expected_rewards
                self.agent.action_function[state[0], state[1], action_index] += ((1 / n) * delta)
                visited_state_actions.add((state, action_index))
            elif (state, action_index) in visited_state_actions and first_visit:
                continue

        # We could update the policy, than we perform on-policy MC control.
        # if we do not update the policy we can do off-policy learning.
        # self.agent.policy = self.agent.action_function.copy()

    def monte_carlo_evaluation(self, epsilon=0.1, gamma=0.9, first_visit=True):
        """
        Find the value function of states using MC evaluation
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param first_visit: Boolean, count only from first occurence of state
        :return:
        """
        state = (0, 0)
        self.env.state = state
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        discounts =  np.array([pow(gamma,i) for i in range(0, len(rewards))])
        visited_states = set()
        for idx, state in enumerate(states):
            if state not in visited_states and first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(discounts[:(len(rewards) - idx)] * rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif not first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(discounts[:(len(rewards) - idx)] * rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif state in visited_states and first_visit:
                continue

    def TD_zero(self, epsilon=0.1, gamma=0.9, alpha=0.05, max_steps=1000):
        """
        Find the value function of move chess states
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param alpha: learning rate
        :param max_steps: max amount of steps in an episode
        """
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            self.agent.value_function[state[0], state[1]] = self.agent.value_function[state[0], state[1]] + alpha * (
                    reward + gamma * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0], state[1]])
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def TD_lambda(self, epsilon=0.1, gamma=0.9, alpha=0.05, lamb=0.9, max_steps=1000):
        """
        Find the value function of move chess states
        :param epsilon: exploration rate
        :param gamma: discount factor of future rewards 
        :param alpha: learning rate
        :param lamb: lambda parameter describing the decay over n-step returns
        :param max_steps: max amount of steps in an episode
        """
        self.agent.E = np.zeros(self.agent.value_function.shape)
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        count_steps = 0
        episode_end = False
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            delta = reward + gamma * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0],
                state[1]]
            self.agent.E[state[0], state[1]] += 1

            # Note to self: vectorize code below.
            self.agent.value_function = self.agent.value_function + alpha * delta * self.agent.E
            self.agent.E = gamma * lamb * self.agent.E
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def evaluate_state_no_policy(self, state, gamma=0.9, synchronous=True):
        """
        Calculates the value of a state based on the successor states and the immediate rewards.
        Because there is no policy applied the action that results in the maximal state is chosen.
        Args:
            state: tuple of 2 integers 0-7 representing the state
            gamma: float, discount factor
            synchronous: Boolean

        Returns: The expected value of the state. 

        """
        state_values = []
        for i in range(len(self.agent.action_space)): 
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.agent.action_space[i])
            if synchronous:
                successor_state_value = self.agent.value_function_prev[self.env.state]
            else:
                successor_state_value = self.agent.value_function[self.env.state]
            state_values.append(
                    reward + gamma * successor_state_value)  # rewards and discounted successor state value
        return np.max(state_values)

    def evaluate_state(self, state, gamma=0.9, synchronous=True):
        """
        Calculates the value of a state based on the successor states and the immediate rewards.
        Args:
            state: tuple of 2 integers 0-7 representing the state
            gamma: float, discount factor
            synchronous: Boolean

        Returns: The expected value of the state under the current policy.

        """
        greedy_action_value = np.max(self.agent.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.agent.policy[state[0], state[1], :]) if
                          a == greedy_action_value]  # List of all greedy actions
        prob = 1 / len(greedy_indices)  # probability of an action occuring
        state_value = 0
        for i in greedy_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.agent.action_space[i])
            if synchronous:
                successor_state_value = self.agent.value_function_prev[self.env.state]
            else:
                successor_state_value = self.agent.value_function[self.env.state]
            state_value += (prob * (
                    reward + gamma * successor_state_value))  # sum up rewards and discounted successor state value
        return state_value

    def evaluate_policy(self, gamma=0.9, synchronous=True):
        self.agent.value_function_prev = self.agent.value_function.copy()  # For synchronous updates
        for row in range(self.agent.value_function.shape[0]):
            for col in range(self.agent.value_function.shape[1]):
                self.agent.value_function[row, col] = self.evaluate_state((row, col), gamma=gamma,
                                                                          synchronous=synchronous)

    def improve_policy(self):
        """
        Finds the greedy policy w.r.t. the current value function
        """
        self.agent.policy_prev = self.agent.policy.copy()
        for row in range(self.agent.action_function.shape[0]):
            for col in range(self.agent.action_function.shape[1]):
                for action in range(self.agent.action_function.shape[2]):
                    self.env.state = (row, col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.agent.action_space[action])
                    successor_state_value = 0 if episode_end else self.agent.value_function[self.env.state]
                    self.agent.policy[row, col, action] = reward + successor_state_value

                max_policy_value = np.max(self.agent.policy[row, col, :])
                max_indices = [i for i, a in enumerate(self.agent.policy[row, col, :]) if a == max_policy_value]
                for idx in max_indices:
                    self.agent.policy[row, col, idx] = 1

    def policy_iteration(self, eps=0.1, gamma=0.9, iteration=1, k=32, synchronous=True):
        """
        Finds the optimal policy
        Args:
            eps: float, exploration rate
            gamma: float, discount factor
            iteration: the iteration number
            k: (int) maximum amount of policy evaluation iterations
            synchronous: (Boolean) whether to use synchronous are asynchronous back-ups 

        Returns:

        """
        policy_stable = True
        print("\n\n______iteration:", iteration, "______")
        print("\n policy:")
        self.visualize_policy()

        print("")
        value_delta_max = 0
        for _ in range(k):
            self.evaluate_policy(gamma=gamma, synchronous=synchronous)
            value_delta = np.max(np.abs(self.agent.value_function_prev - self.agent.value_function))
            value_delta_max = value_delta
            if value_delta_max < eps:
                break
        print("Value function for this policy:")
        print(self.agent.value_function.round().astype(int))
        action_function_prev = self.agent.action_function.copy()
        print("\n Improving policy:")
        self.improve_policy()
        policy_stable = self.agent.compare_policies() < 1
        print("policy diff:", policy_stable)

        if not policy_stable and iteration < 1000:
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_stable:
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            print("failed to converge.")

    def visualize_policy(self):
        """
        Gives you are very ugly visualization of the policy
        Returns: None

        """
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {}
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]
        elif self.agent.piece == 'bishop':
            arrows = "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'rook':
            arrows = "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col]

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)
