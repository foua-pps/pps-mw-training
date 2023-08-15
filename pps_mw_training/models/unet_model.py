from dataclasses import dataclass
from pathlib import Path
import json

import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.utils.blocks import (
    ConvolutionBlock,
    DownsamplingBlock,
    MLP,
    UpsamplingBlock,
)
from pps_mw_training.utils.data import prepare_dataset
from pps_mw_training.utils.loss_function import quantile_loss


class UNetBaseModel(keras.Model):
    """
    Keras implementation of the UNet architecture, an input block followed
    by 4 encoder blocks and 4 decoder blocks, and finishing with a
    multi layer perceptron block.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_unet_base: int,
        n_features: int,
        n_layers: int,
    ):
        super().__init__()
        n = n_unet_base
        self.in_block = ConvolutionBlock(n_inputs, n)
        self.down_block_1 = DownsamplingBlock(n * 1, n * 2)
        self.down_block_2 = DownsamplingBlock(n * 2, n * 4)
        self.down_block_3 = DownsamplingBlock(n * 4, n * 8)
        self.down_block_4 = DownsamplingBlock(n * 8, n * 16)
        self.up_block_1 = UpsamplingBlock(n * 16, n * 8)
        self.up_block_2 = UpsamplingBlock(n * 8, n * 4)
        self.up_block_3 = UpsamplingBlock(n * 4, n * 2)
        self.up_block_4 = UpsamplingBlock(n * 2, n * 1)
        self.out_block = MLP(n_outputs, n_features, n_layers)

    def call(self, inputs):
        d_0 = self.in_block(inputs)
        d_1 = self.down_block_1(d_0)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)
        u = self.up_block_1([d_4, d_3])
        u = self.up_block_2([u, d_2])
        u = self.up_block_3([u, d_1])
        u = self.up_block_4([u, d_0])
        return self.out_block(u)

    def build_graph(self, image_size: int, n_inputs: int):
        x = keras.Input(shape=(image_size, image_size, n_inputs))
        return keras.Model(inputs=[x], outputs=self.call(x))


@dataclass
class UNetModel:
    """Unet model object."""
    model: UNetBaseModel

    @classmethod
    def load(
        cls,
        model_config_file: Path,
    ) -> "UNetModel":
        """Load the model from config file."""
        with open(model_config_file) as config_file:
            config = json.load(config_file)
        n_inputs = config["n_inputs"]
        n_outputs = len(config["quantiles"])
        model = UNetBaseModel(
            n_inputs,
            n_outputs,
            config["n_unet_base"],
            config["n_features"],
            config["n_layers"],
        )
        model.build((None, None, None, n_inputs))
        model.load_weights(config["model_weights"])
        return cls(model)

    @staticmethod
    def train(
        n_inputs: int,
        n_unet_base: int,
        n_features: int,
        n_layers: int,
        quantiles: list[float],
        training_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        batch_size: int,
        n_epochs: int,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float,
        m_mul: float,
        alpha: float,
        output_path: Path,
    ) -> None:
        """Train the model."""
        n_outputs = len(quantiles)
        model = UNetBaseModel(
            n_inputs, n_outputs, n_unet_base, n_features, n_layers,
        )
        model.build((None, None, None, n_inputs))
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: quantile_loss(
                1, quantiles, y_true, y_pred,
            ),
        )
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )
        weights_file = output_path / "weights.h5"
        history = model.fit(
            prepare_dataset(training_data, n_inputs, batch_size, augment=True),
            epochs=n_epochs,
            validation_data=prepare_dataset(
                validation_data,
                n_inputs,
                batch_size,
                augment=True,
            ),
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    weights_file,
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
        with open(output_path / "fit_history.json", "w") as outfile:
            outfile.write(json.dumps(history.history, indent=4))
        with open(output_path / "network_config.json", "w") as outfile:
            outfile.write(
                json.dumps(
                    {
                        "n_inputs": n_inputs,
                        "n_unet_base": n_unet_base,
                        "n_features": n_features,
                        "n_layers": n_layers,
                        "quantiles": quantiles,
                        "model_weights": weights_file.as_posix(),
                    },
                    indent=4,
                )
            )

    def predict(
        self,
        input_data: tf.data.Dataset,
    ) -> tf.data.Dataset:
        """Apply the trained neural network for a retrieval purpose."""
        return self.model.predict(input_data)
