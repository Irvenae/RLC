from ray import tune
from ray.rllib.agents.trainer import with_common_config
# Restructuring in the docs, new release (0.8.5) will have exection folder
# from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
# from ray.rllib.execution.train_ops import TrainOneStep
# from ray.rllib.execution.metric_ops import StandardMetricsReporting
# currently this lives in experimental
from ray.rllib.utils.experimental_dsl import ParallelRollouts, ConcatBatches, TrainOneStep, StandardMetricsReporting
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.rollout import create_parser, run

from RLC.capture_chess_rllib.policies import PolicyGradient, PolicyRandom
from RLC.capture_chess_rllib.environment import CaptureChessEnv
from RLC.capture_chess_rllib.models import register_models
from RLC.utils import dotdict

# Define a trainer using our own defined policy.
# The trainer will run multiple workers and update the policy given by the ruling underneath.
# (this is the standard PG as implemented in RLLib)
# Evaluation of a policy can be done with the function rollout in rollout.py.
# https://github.com/ray-project/ray/blob/master/rllib/examples/rollout_worker_custom_workflow.py


def execution_plan(workers, config):
    # Experimental distributed execution impl; enable with "use_exec_api": True.
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


# Because we need to set the setting to True on default, we will override the parameters.
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default.
    "num_workers": 0,
    # Learning rate.
    "lr": 0.0004,
    # Use the execution plan API instead of policy optimizers.
    "use_exec_api": True,
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


def play_random_games(n_episodes):
    """
    Plays a number of games against a player playing at random.
    Expect in the episode_reward always a value of 0.0 because sum over all players should always be zero.
    (when the reward is positive for player1 it is negative for player2)
    """
    # register the models that can be used in the config.
    register_models()

    def select_policy(agent_id):
        if agent_id == "player1":
            return "learned"
        else:
            return "random"

    config = {
        "env": CaptureChessEnv,
        "gamma": 0.9,
        "lr": 0.1,
        "num_workers": 0,   # Will spin up cores = num_workers + 1
        # "num_envs_per_worker": 4,
        "rollout_fragment_length": 100,  # Fragment to run in worker.
        "train_batch_size": 100,    # size of a batch for training in steps.
        "batch_mode": "complete_episodes",
        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
                "random": (PolicyRandom, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
                           {}),
                # learned will use the default policy of the trainer (PolicyGradient).
                "learned": (None, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
                            {
                                # model needs to be set here because otherwise we will tie the policy to the environment and the model.
                                "model": {
                                    "custom_model": "pg_model",
                                }
                            }),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    # use restore with path to continue training from previous.
    tune.run(PGTrainer, checkpoint_freq=1, checkpoint_at_end=True,
             stop={"episodes_total": n_episodes}, config=config)


def load_and_evaluate(dir_checkpoint, n_episodes):
    """
    Load a previously stored training session and evaluate.
    e.g. dir_checkpoint: "/Users/irvenaelbrecht/ray_results/PolicyGradientTrainer/PolicyGradientTrainer_CaptureChessEnv_0_2020-05-07_08-41-44p2dp1i_z/checkpoint_1/checkpoint-1"

    Args:
        dir_checkpoint
            directory to load from
        n_episodes
            number of episodes to evaluate
    """
    # register the models that can be used in the config.
    register_models()
    tune.register_trainable("PolicyGradientTrainer", PGTrainer)

    parser = create_parser()
    args = dotdict({"run": "PolicyGradientTrainer", "episodes": n_episodes,
                    "checkpoint": dir_checkpoint, "steps": 0, "no_render": True, "config": {}})
    run(args, parser)
