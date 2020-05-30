from functools import partial
import numpy as np
import collections
from scipy.stats import binom_test
import copy
import math
import os

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.rollout import DefaultMapping
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_episodes
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.space_utils import flatten_to_single_ndarray
from ray.tune.registry import get_trainable_cls

from RLC.capture_chess_rllib.policies import PolicyGradient, PolicyRandom
from RLC.capture_chess_rllib.environment import CaptureChessEnv
from RLC.utils import dotdict

# Define a trainer using our own defined policy.
# The trainer will run multiple workers and update the policy given by the ruling underneath.
# (this is the standard PG as implemented in RLLib)
# Evaluation of a policy can be done with the function rollout in rollout.py.
# https://github.com/ray-project/ray/blob/master/rllib/examples/rollout_worker_custom_workflow.py


def execution_plan(workers, config):
    # enable with "use_exec_api": True.
    # Collects experiences in parallel from multiple RolloutWorker actors.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Combine experiences batches until we hit `train_batch_size` in size.
    # Then, train the policy on those experiences and update the workers.
    train_op = rollouts \
        .combine(ConcatBatches(
            min_batch_size=config["train_batch_size"])) \
        .for_each(TrainOneStep(workers))

    # Add on the standard episode reward, etc. metrics reporting. This returns
    # a LocalIterator[metrics_dict] representing metrics for each train step.
    return StandardMetricsReporting(train_op, workers, config)


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env,
                       policies, episode,
                       **kwargs):
        episode.custom_metrics["winner"] = base_env.get_unwrapped()[
            0].determine_winner()


# Because we need to set the setting to True on default, we will override the parameters.
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default.
    "num_workers": 0,
    # Learning rate.
    "lr": 0.0004,
    # Use the execution plan API instead of policy optimizers.
    "use_exec_api": True,
    "callbacks": MyCallbacks,
})

# Define the trainer.
# From the _setup() function in trainer.py, we can see how the env is setup.
# The main function is _train() in trainer_template.py.
# Here we can see how the execution_plan or other training is called.
PGTrainer = build_trainer(
    name="PolicyGradientTrainer",
    default_config=DEFAULT_CONFIG,
    default_policy=PolicyGradient,
    execution_plan=execution_plan,
)


class InfoNumberRounds():
    def __init__(self, min_, max_, step):
        self.min = min_
        self.max = max_
        self.step = step


def self_play_workflow(config):
    """
    Expects in config:
        checkpoint
            checkpoint to load from (None if new)
        trainer
            trainer to use
        model
            model to use in learning
        percentage_equal: float
            The maximal allowed percentage that equal opponents get game results. (see binomial test)
        lr_schedule: List of lr
            Learning rates to use. Will use first to last and update each time the model gets worse.
       training_rounds
            Rounds of training
        evaluation_rounds
            Rounds of evaluation

    1. Generate a large batch of self-play games.
    2. Train.
    3. Test the updated bot against the previous version.
    4. If the bot is measurably stronger, switch to this new version.
    5. If the bot is about the same strength, generate more games and train again.
    6. If the bot gets significantly weaker, adjust the optimizer settings and retrain.
    """
    ##########################################
    # Set config of trainer and evaluators
    ##########################################
    check_dir = 'logs'
    log_file = 'logs/logs.txt'
    if os.path.exists(log_file):
        os.remove(log_file)

    if config.get("evaluation_num_episodes", None) is None:
        config["evaluation_num_episodes"] = 1
    trainer_fn = get_trainable_cls(config["trainer"])
    lr_idx = 0

    def select_policy_train(agent_id):
        if agent_id == "player1":
            return np.random.choice(["learning_white", "previous_white", "random"], 1,
                                    p=[.6, .3, .1])[0]
        else:
            return np.random.choice(["learning_black", "previous_black", "random"], 1,
                                    p=[.6, .3, .1])[0]

    def select_policy_eval(learning_player, agent_id):
        if learning_player == "player1":
            if agent_id == "player1":
                return "learning_white"
            else:
                return "previous_black"
        else:
            if agent_id == "player2":
                return "learning_black"
            else:
                return "previous_white"

    trainer_config = copy.deepcopy(config)
    # remove self-play parameters
    trainer_config.pop("trainer")
    trainer_config.pop("percentage_equal")
    trainer_config.pop("model")
    trainer_config.pop("training_rounds")
    trainer_config.pop("evaluation_rounds")
    trainer_config.pop("checkpoint", None)
    trainer_config.pop("lr_schedule", None)
    trainer_config.pop("evaluation_interval", None)

    trainer_config["lr"] = config["lr_schedule"][lr_idx]
    trainer_config["multiagent"] = {
        "policies_to_train": ["learning_white", "learning_black"],
        "policies": {
            "random": (PolicyRandom, config["env"].observation_space, config["env"].action_space,
                       {}),
            "learning_white": (None, config["env"].observation_space, config["env"].action_space,
                               {
                "model": config["model"]
            }),
            "learning_black": (None, config["env"].observation_space, config["env"].action_space,
                               {
                "model": config["model"]
            }),
            "previous_white": (None, config["env"].observation_space, config["env"].action_space,
                               {
                "model": config["model"]
            }),
            "previous_black": (None, config["env"].observation_space, config["env"].action_space,
                               {
                "model": config["model"]
            }),
        },
        "policy_mapping_fn": select_policy_train,
    }
    trainer_config["train_batch_size"] = 2 * config["train_batch_size"]

    eval_config_player1 = copy.deepcopy(trainer_config)
    eval_config_player1["multiagent"]["policy_mapping_fn"] = partial(
        select_policy_eval, "player1")
    eval_config_player1["multiagent"]["policies_to_train"] = []

    eval_config_player2 = copy.deepcopy(trainer_config)
    eval_config_player2["multiagent"]["policy_mapping_fn"] = partial(
        select_policy_eval, "player2")
    eval_config_player2["multiagent"]["policies_to_train"] = []

    ##########################################
    # Run train / evaluation rounds
    ##########################################

    def update_for_next_loop(total_rounds, rounds, reset=False):
        done = False
        if reset:
            next_num_rounds = rounds.min
        else:
            if (total_rounds >= rounds.max):
                done = True
            next_num_rounds = rounds.step

        return done, next_num_rounds

    ray.init()

    trainer = trainer_fn(
        env=trainer_config["env"], config=trainer_config)
    evaluator_player1 = trainer_fn(
        env=eval_config_player1["env"], config=eval_config_player1)
    evaluator_player2 = trainer_fn(
        env=eval_config_player1["env"], config=eval_config_player2)

    total_rounds_training = 0
    done, training_rounds = update_for_next_loop(
        total_rounds_training, config["training_rounds"], True)
    prev_it_state = config.get("checkpoint", None)
    prev_state = prev_it_state
    while not done:
        ##########################################
        # Train
        ##########################################
        try:
            if prev_it_state is not None:
                trainer.restore(prev_it_state)
            for _ in range(training_rounds):
                trainer.train()
            state = trainer.save(check_dir)
            # trainer.stop()

            total_rounds_training += training_rounds
        except Exception:
            trainer.stop()
            with open(log_file, 'a') as f:
                f.write("Model failed, updating optimizer\n")
            lr_idx += 1
            if lr_idx < len(config["lr_schedule"]):
                trainer_config["lr"] = config["lr_schedule"][lr_idx]
                trainer = trainer_fn(
                    env=trainer_config["env"], config=trainer_config)
                total_rounds_training = 0
                done, training_rounds = update_for_next_loop(
                    total_rounds_training, config["training_rounds"], True)
                prev_it_state = prev_state
            else:
                done = True
            continue  # try again.

        ##########################################
        # Evaluate
        ##########################################
        try:
            total_eval_rounds = 0
            comparison_wrt_equal = 1
            eval_results1 = []
            eval_results2 = []
            # maximal evaluation rounds determined by training, does not make sense to evaluate more than training rounds.
            eval_info = InfoNumberRounds(config["evaluation_rounds"].min, min(
                config["evaluation_rounds"].max, total_rounds_training), config["evaluation_rounds"].step)
            done_eval, eval_rounds = update_for_next_loop(
                total_eval_rounds, eval_info, True)
            while not done_eval:
                num_episodes = eval_rounds * config["evaluation_num_episodes"]

                evaluator_player1.restore(state)
                eval_results1.extend(own_evaluation(
                    evaluator_player1, eval_rounds))
                num_pos = sum(x == 1 for x in eval_results1)
                num_neg = sum(x == -1 for x in eval_results1)
                comparison_wrt_equal1 = binom_test(
                    num_pos, num_pos + num_neg, 0.5)
                with open(log_file, 'a') as f:
                    f.write(
                        f'results1: trained agent wins: {num_pos} previous agent wins: {num_neg} remises: {sum(x == 0 for x in eval_results1)} \n')
                    f.write(
                        f'chance result for equal opponents: {comparison_wrt_equal1} \n')

                evaluator_player2.restore(state)
                eval_results2.extend(own_evaluation(
                    evaluator_player2, eval_rounds))
                num_pos = sum(x == 1 for x in eval_results2)
                num_neg = sum(x == -1 for x in eval_results2)
                comparison_wrt_equal2 = binom_test(
                    num_neg, num_pos + num_neg, 0.5)
                with open(log_file, 'a') as f:
                    f.write(
                        f'results2: trained agent wins: {num_neg} previous agent wins: {num_pos} remises: {sum(x == 0 for x in eval_results2)} \n')
                    f.write(
                        f'chance result for equal opponents: {comparison_wrt_equal2} \n')

                total_eval_rounds += eval_rounds

                done_eval, eval_rounds = update_for_next_loop(
                    total_eval_rounds, eval_info)
                if config["percentage_equal"] > comparison_wrt_equal1 or config["percentage_equal"] > comparison_wrt_equal2:
                    # one of players improved
                    done_eval = True
        except Exception:
            with open(log_file, 'a') as f:
                f.write("Model failed, need to update optimizer\n")
            # trigger update optimizer
            comparison_wrt_equal1 = 0
            comparison_wrt_equal2 = 0
            eval_results1 = [-1]
            eval_results2 = [1]

        ##########################################
        # Update policy
        ##########################################

        if config["percentage_equal"] > comparison_wrt_equal1 or config["percentage_equal"] > comparison_wrt_equal2:
            # results differ enough
            if sum(x == 1 for x in eval_results1) > sum(x == -1 for x in eval_results1) and sum(x == -1 for x in eval_results2) > sum(x == 1 for x in eval_results2):
                with open(log_file, 'a') as f:
                    f.write("Model improved\n")
                total_rounds_training = 0
                done, training_rounds = update_for_next_loop(
                    total_rounds_training, config["training_rounds"], True)
                # reupdate previous
                key_previous_val_learning_white = {}
                for (k, v), (k2, v2) in zip(trainer.get_policy("previous_white").get_weights().items(),
                                            trainer.get_policy("learning_white").get_weights().items()):
                    key_previous_val_learning_white[k] = v2
                key_previous_val_learning_black = {}
                for (k, v), (k2, v2) in zip(trainer.get_policy("previous_black").get_weights().items(),
                                            trainer.get_policy("learning_black").get_weights().items()):
                    key_previous_val_learning_black[k] = v2
                # set weights
                trainer.set_weights({"previous_white": key_previous_val_learning_white,
                                     "previous_black": key_previous_val_learning_black,
                                     # no change
                                     "learning_white": trainer.get_policy("learning_white").get_weights(),
                                     "learning_black": trainer.get_policy("learning_black").get_weights(),
                                     })
                if prev_state is not None:
                    trainer.delete_checkpoint(prev_state)
                trainer.delete_checkpoint(state)

                prev_it_state = trainer.save(check_dir)
                prev_state = prev_it_state
            elif sum(x == 1 for x in eval_results1) < sum(x == -1 for x in eval_results1) and sum(x == -1 for x in eval_results2) < sum(x == 1 for x in eval_results2):
                with open(log_file, 'a') as f:
                    f.write("Model got worse, updating optimizer\n")
                trainer.stop()
                lr_idx += 1
                if lr_idx < len(config["lr_schedule"]):
                    trainer_config["lr"] = config["lr_schedule"][lr_idx]
                    trainer = trainer_fn(
                        env=trainer_config["env"], config=trainer_config)
                    total_rounds_training = 0
                    done, training_rounds = update_for_next_loop(
                        total_rounds_training, config["training_rounds"], True)
                    prev_it_state = prev_state
                else:
                    done = True
            else:
                with open(log_file, 'a') as f:
                    f.write(
                        "One player improved one got worse, trying more learning iterations.\n")
                done, training_rounds = update_for_next_loop(
                    total_rounds_training, config["training_rounds"])
                prev_it_state = state
        else:
            with open(log_file, 'a') as f:
                f.write("Unable to evaluate, trying more learning iterations.\n")
            done, training_rounds = update_for_next_loop(
                total_rounds_training, config["training_rounds"])
            prev_it_state = state

    trainer.restore(prev_it_state)
    trainer.save()
    print("Checkpoint and trainer saved at: ", trainer.logdir)
    with open(log_file, 'a') as f:
        f.write(f'Checkpoint and trainer saved at: {trainer.logdir} \n')


def own_rollout(agent,
                num_episodes):
    results = []
    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    else:
        raise NotImplementedError("Multi-Agent only")

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    episodes = 0
    while episodes < num_episodes:
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        while not done and (episodes < num_episodes):
            action_dict = {}
            for agent_id, a_obs in obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))

                    a_action = agent.compute_action(
                        a_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action

            action = action_dict
            next_obs, reward, done, info = env.step(action)
            done = done["__all__"]

            # update
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r
            obs = next_obs

        if done:
            episodes += 1
            # specific function for alternate game.
            results.append(env.determine_winner())

    return results


def own_evaluation(agent, num_rounds):
    results = []
    num_episodes = num_rounds * agent.config["evaluation_num_episodes"]
    if agent.config["num_workers"] == 0:
        for _ in range(num_episodes):
            agent.evaluation_workers.local_worker().sample()
    else:
        while len(results) < num_episodes:
            # Calling .sample() runs exactly one episode per worker due to how the
            # eval workers are configured.
            ray.get([
                w.sample.remote()
                for w in agent.workers.remote_workers()
            ])

            episodes, _ = collect_episodes(None,
                                           agent.workers.remote_workers(),
                                           [])

            for episode in episodes:
                for key, winner in episode.custom_metrics.copy().items():
                    results.append(winner)

    return results[:num_episodes]
