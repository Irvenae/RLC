import keras.backend.tensorflow_backend as tfback
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

from keras.models import Model, clone_model, load_model
from keras.layers import Input, Conv2D, Dense, Reshape, Dot, Activation, Multiply, concatenate

tf = try_import_tf()

# start hotfix module 'tensorflow._api.v2.config' has no attribute 'experimental_list_devices'


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


def register_models():
    ModelCatalog.register_custom_model("pg_model", PolicyGradientModel)


class PolicyGradientModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(PolicyGradientModel, self).__init__(obs_space, action_space,
                                                  num_outputs, model_config, name)
        input_layer = Input(shape=[8,8,8], name='board_layer')  # 8,8,8
        legal_moves = Input(shape=action_space.shape, name='legal_move_mask')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(
            input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(
            input_layer)  # 1,8,8
        inter_layer_3 = Conv2D(1, (1, 1), data_format="channels_first")(
            input_layer)  # 1,8,8
        inter_layer_4 = Conv2D(1, (1, 1), data_format="channels_first")(
            input_layer)  # 1,8,8
        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
        flat_3 = Reshape(target_shape=(1, 64))(inter_layer_3)
        flat_4 = Reshape(target_shape=(1, 64))(inter_layer_4)
        output_dot_layer1 = Dot(axes=1)([flat_1, flat_2])
        output_dot_layer2 = Dot(axes=1)([flat_3, flat_4])
        output_action_probs = Reshape(target_shape=action_space.shape)(output_dot_layer1)
        output_action_var = Reshape(target_shape=action_space.shape)(output_dot_layer2)
        legal_output_action_probs = Multiply()(
            [legal_moves, output_action_probs])  # Select legal moves
        legal_output_action_var = Multiply()(
            [legal_moves, output_action_var])  # Select legal moves
        output_layer = concatenate([legal_output_action_probs, legal_output_action_var])
        # output layer is normalized by action_distribution (see loss function).
        value_out = Dense(1, name="value_out", activation=None)(output_layer)
        self.base_model = Model(inputs=[input_layer, legal_moves], outputs=[
                                output_layer, value_out])
        # self.base_model.summary()
        self.register_variables(self.base_model.weights)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(
            [input_dict["obs"]["real_obs"], input_dict["obs"]["action_mask"]])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {}
