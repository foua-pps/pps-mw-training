from pathlib import Path
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
import pickle
from pps_mw_training.pipelines.cloud_base import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor


VMIN = -30
VMAX = 60
CMAP = "coolwarm"
QUANTILE_IDXS = [3, 4, 5]
DBZ_MIN = 15


def evaluate_model(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
    output_path: Path,
) -> None:
    """Evaluate model, very rudimentary for the moment."""
    predictor.model.summary()
    keras.utils.plot_model(
        predictor.model.build_graph(settings.IMAGE_SIZE, len(settings.INPUT_PARAMS)),
        to_file=settings.MODEL_CONFIG_PATH / "model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
    )
    stats = get_stats(predictor, test_data)
    with open(output_path / "retrieval_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))
    # plot_stats(predictor, test_data)


def get_stats(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
) -> dict[str, float]:
    """Get stats."""
    preds = []
    labels = []
    images = []

    for input_data, label_data in test_data:
        print(input_data.numpy().shape)
        for p in predictor.model(input_data).numpy():
            preds.append(p)
        for r in label_data.numpy():
            labels.append(r)
        for i in input_data.numpy():
            #i_full = np.full(label_data.numpy()[0].shape, settings.FILL_VALUE_IMAGES)
            images.append(i)

    preds = np.expand_dims(
        np.array(preds)[:, :, :, len(settings.QUANTILES) // 2], axis=3
    )
    labels = np.array(labels)
    images = np.array(images)
    
    print(labels.shape, images.shape)

    # print(labels.shape, images.shape)
    # filt = (labels != settings.FILL_VALUE_LABELS) & (
    #     images != settings.FILL_VALUE_IMAGES) & np.expand_dims(np.all(input_data.numpy() > 0, axis=-1), axis=3)
    # fig, ax = plt.subplots(2, 2)
    # ax = ax.ravel()
    # ix = 86
    # cs = ax[0].pcolormesh(labels[ix, :, :, 0], vmin=500, vmax=1000, cmap="tab20c")
    # plt.colorbar(cs, ax=ax[0])
    # cs = ax[1].pcolormesh(preds[ix, :, :, 0], vmin=500, vmax=1000, cmap="tab20c")
    # plt.colorbar(cs, ax=ax[1])
    # cs = ax[2].pcolormesh(input_data[ix, :, :, 2], cmap="tab20c")
    # plt.colorbar(cs, ax=ax[2])
    # cs = ax[3].pcolormesh(input_data[ix, :, :, 1], cmap="tab20c")
    # plt.colorbar(cs, ax=ax[3])
    # fig.savefig("try.png")
    
    with open("predictions.pickle", "wb") as f:
        print(labels.shape, images.shape, preds.shape)
        pickle.dump(labels, f)
        pickle.dump(images, f)
        pickle.dump(preds, f)
        
    #print(float(np.sqrt(np.nanmean((preds[filt] - labels[filt]) ** 2))))

    # return {
    #     "rmse": float(np.sqrt(np.nanmean((preds - labels) ** 2))),
    #     "corr": float(np.corrcoef(preds, labels)[0, 1]),
    # }


# def plot_stats(
#     predictor: UnetPredictor,
#     test_data: tf.data.Dataset,
# ) -> None:
#     """Plot stats."""
#     for input_data, label_data in test_data:
#         pred = predictor.model(input_data).numpy()
#         for idx in range(pred.shape[0]):
#             plt.figure()
#             plt.subplot(2, 3, 1)
#             plt.imshow(input_data[idx, :, :, 0], cmap=CMAP)
#             plt.colorbar()
#             plt.title("MW_160")
#             plt.subplot(2, 3, 2)
#             plt.imshow(mw_data[idx, :, :, -1], cmap=CMAP)
#             plt.colorbar()
#             plt.title("MW_183")
#             plt.subplot(2, 3, 3)
#             plt.imshow(radar_data[idx], cmap=CMAP, vmin=VMIN, vmax=VMAX)
#             plt.colorbar()
#             plt.title("Radar")
#             for i, qidx in enumerate(QUANTILE_IDXS):
#                 plt.subplot(2, 3, 4 + i)
#                 plt.imshow(
#                     pred[idx, :, :, qidx],
#                     cmap=CMAP,
#                     vmin=VMIN,
#                     vmax=VMAX,
#                 )
#                 plt.title(f"Quantile: {settings.QUANTILES[qidx]}")
#                 plt.colorbar()
#             plt.show()
