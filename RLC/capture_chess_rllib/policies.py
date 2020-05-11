from ray.rllib.evaluation.postprocessing import discount, Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
# Wrapper around a dictionary with string keys and array-like values.
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf

import keras.backend as K
import numpy as np
import random

tf = try_import_tf()


def policy_gradient_loss(policy, model, dist_class, train_batch):
    """
    Policy gradient loss is defined as the sum, over timesteps, of action
    log-probability (same as categorical_crossentropy) times the cumulative rewards from that timestep onward.

    Args:
        train_batch: dict
            contains all the inputs
    """
    # Call model with a batch of data (will return outputs as first argument, second argument is state).
    logits, _ = model.from_batch(train_batch)
    # Normalize output model by distribution.
    action_dist = dist_class(logits, model)
    # call softmax_cross_entropy_with_logits
    action_cross_entropy = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return - K.mean(action_cross_entropy * train_batch[Postprocessing.ADVANTAGES])


def calculate_advantages(policy,
                         sample_batch,
                         other_agent_batches=None,
                         episode=None):
    sample_batch[Postprocessing.ADVANTAGES] = discount(
        sample_batch[SampleBatch.REWARDS],  policy.config["gamma"])
    return sample_batch

# We make a policy with gradients using the PolicyGradientModel.
# For this we define a specific policy gradient loss.
# We also convert the rewards to returns by calculate_advantages.
# The method learn_on_batch will update the model when the trainer calls this.
PolicyGradient = build_tf_policy(
    name="PolicyGradient",
    loss_fn=policy_gradient_loss,
    postprocess_fn=calculate_advantages,)
    # If no optimizer is given, the Adam optimizer is used.
    # Difficult to get keras optimizer working because of no compute_gradients() function.
    # You would need to use a gradient tape for this where the model is called.


class PolicyRandom(Policy):
    """Play a random move."""

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # Dictionary of obs is converted to np array and batched by adding it to a list.
        chosen_actions = np.zeros(shape=(len(obs_batch), 4096))
        for i, obs in enumerate(obs_batch):
            action_mask = obs[513:]
            position_actions = np.where(action_mask == 1)
            chosen_action_int = random.randint(0, len(position_actions) - 1)
            chosen_actions[i][position_actions[chosen_action_int]] = 1

        return chosen_actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
