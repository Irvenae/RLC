from ray import tune
from ray.rllib.rollout import create_parser, run

from RLC.capture_chess_rllib.policies import PolicyGradient, PolicyRandom
from RLC.capture_chess_rllib.environment import CaptureChessEnv
from RLC.capture_chess_rllib.models import register_models
from RLC.capture_chess_rllib.game_internal import PGTrainer, InfoNumberRounds, self_play_workflow
from RLC.utils import dotdict


def self_play(trainer="PolicyGradientTrainer", restore_path = None):
    # TODO: Change name model dependent on used trainer.
    register_trainers()

    episode_length = 100
    num_episodes_round = 500 # used for each side.
    config = {
        "env": CaptureChessEnv,
        "gamma": 0.95,
        "num_workers": 20,   # Will spin up cores = num_workers + 1
        "num_envs_per_worker": 4,
        # Fragment to run in worker.
        "rollout_fragment_length": 10 * episode_length,
        # size of a batch for training in steps.
        "train_batch_size": num_episodes_round * episode_length,
        "batch_mode": "complete_episodes",
        "checkpoint": restore_path,
        # specifics for self-play
        "trainer": trainer,
        "model": { "custom_model": "pg_model"},
        "lr_schedule":[0.3, 0.1, 0.01, 0.001, 0.0001],
        "evaluation_num_episodes": num_episodes_round,
        "percentage_equal": 0.05,
        "training_rounds": InfoNumberRounds(1, 100, 10),
        "evaluation_rounds": InfoNumberRounds(1, 50, 5),
    }

    # use restore with path to continue training from previous.
    self_play_workflow(config)


def play_random_games(n_training_rounds, trainer="PolicyGradientTrainer"):
    """
    Plays a number of games against a player playing at random.
    Expect in the episode_reward always a value of 0.0 because sum over all players should always be zero.
    (when the reward is positive for player1 it is negative for player2)
    """
    # TODO: Change name model dependent on used trainer.
    # register the models that can be used in the config.
    register_trainers()

    def select_policy(agent_id):
        if agent_id == "player2":
            return "learning"
        else:
            return "random"

    episode_length = 100
    config = {
        "env": CaptureChessEnv,
        "gamma": 0.9,
        "lr": 0.3,
        "num_workers": 3,   # Will spin up cores = num_workers + 1
        "num_envs_per_worker": 4,
        # Fragment to run in worker.
        "rollout_fragment_length": 10 * episode_length,
        # size of a batch for training in steps.
        "train_batch_size": 500 * episode_length,
        "batch_mode": "complete_episodes",
        "multiagent": {
            "policies_to_train": ["learning"],
            "policies": {
                "random": (PolicyRandom, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
                           {}),
                # learned will use the default policy of the trainer (PolicyGradient).
                "learning": (None, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
                             {
                                 # model needs to be set here because otherwise we will tie the policy to the environment and the model.
                                 "model": {
                                     "custom_model": "pg_model",
                                 }
                             }),
            },
            "policy_mapping_fn": select_policy,
        },
        # # Enable evaluation, once per training iteration.
        # "evaluation_interval": 1,

        # # Run 10 episodes each time evaluation runs.
        # "evaluation_num_episodes": 100,
    }
    # use restore with path to continue training from previous.
    tune.run(trainer, checkpoint_at_end=True,
             stop={"training_iteration": n_training_rounds}, config=config)


def load_and_evaluate(dir_checkpoint, n_episodes, video_dir=None):
    """
    Load a previously stored training session and evaluate.
    e.g. dir_checkpoint: "/Users/irvenaelbrecht/ray_results/PolicyGradientTrainer/PolicyGradientTrainer_CaptureChessEnv_0_2020-05-07_08-41-44p2dp1i_z/checkpoint_1/checkpoint-1"

    Args:
        dir_checkpoint
            directory to load from
        n_episodes
            number of episodes to evaluate
        video_dir
            directory to save video of episode, currently this can only be done by
            disabling some of the stats_recorder:
                - disable checking functionality in before_step(self, action):
                - self.rewards += reward -> self.rewards += 0 in after_step(self, observation, reward, done, info)
    """
    # register the models that can be used in the config.
    register_trainers()

    def select_policy(agent_id):
        if agent_id == "player1":
            return "learning_white"
        else:
            return "learning_black"

    parser = create_parser()
    config = {"multiagent": {
        "policies_to_train": [],
        "policies": {
            # "random": (PolicyRandom, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
            #            {}),
            # learned will use the default policy of the trainer (PolicyGradient).
            "learning_white": (None, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
                         {
                             # model needs to be set here because otherwise we will tie the policy to the environment and the model.
                             "model": {
                                 "custom_model": "pg_model",
                             }
                         }),
            "learning_black": (None, CaptureChessEnv.observation_space, CaptureChessEnv.action_space,
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
    args = dotdict({"run": "PolicyGradientTrainer", "episodes": n_episodes,
                    "checkpoint": dir_checkpoint, "steps": 0, "no_render": True, "workers": None, "video_dir": video_dir, "config": config})
    run(args, parser)


def register_trainers():
    # register models so we can use name.
    register_models()
    tune.register_trainable("PolicyGradientTrainer", PGTrainer)
