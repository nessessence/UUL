# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
from . import distance
from . import embedding
from . import io_util
import numpy as np


_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)


import os
import numpy as np
import datetime
import glob
from natsort import natsorted

def get_file_creation_date(filepath):
    timestamp = os.path.getctime(filepath)
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%d-%m-%y_%H:%M")

def compute_cmmd(
    ref_dir,
    eval_dir,
    ref_embed_file=None,
    batch_size=32,
    max_count=-1,
    use_precompute_features_if_exist=False,
    model = None,
):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory.
      use_precompute_features_if_exist: Whether to use and save precomputed features.
      n_max_gen_img: Used to name the saved .npy files for caching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")

    if model is None:
        embedding_model = embedding.ClipEmbeddingModel()
    else:
        embedding_model = model

    def get_or_compute_embeddings(base_dir):
        # Find the first image in the folder using natural sort
        image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(base_dir, ext)))

        if not image_files:
            raise FileNotFoundError(f"No image files found in {base_dir}")

        first_image_path = natsorted(image_files)[0]
        date_str = get_file_creation_date(first_image_path)



        feature_dir = os.path.join(base_dir, "precomputed_features", "clip-l-14")
        # feature_dir = os.path.join(base_dir, "precomputed_features", "clip-b-32")
        feature_filename = f"{date_str}_n{max_count}.npy"
        feature_filename = feature_filename # Replace ':' with '-' for filename compatibility
        # feature_dir = feature_dir.replace("+", "--")  # Replace ':' with '-' for filename compatibility
        feature_path = os.path.join(feature_dir, feature_filename)
        # feature_path = feature_path.replace("+", "--")  # Replace ':' with '-' for filename compatibility
        if use_precompute_features_if_exist:

            if os.path.exists(feature_path):
                print(f"Loading precomputed features from {feature_path}")
                return np.load(feature_path).astype("float32")

        # Compute features
        embs = io_util.compute_embeddings_for_dir(base_dir, embedding_model, batch_size, max_count).astype("float32")

        # if use_precompute_features_if_exist:
        os.makedirs(feature_dir, exist_ok=True)
        print(f"Saving computed features to {feature_path}")
        np.save(feature_path, embs)

        return embs

    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = get_or_compute_embeddings(ref_dir)

    eval_embs = get_or_compute_embeddings(eval_dir)

    # print(type(eval_embs))  # <class 'numpy.ndarray'>
    # print(eval_embs.shape)  # e.g., (50, 768)

    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


# def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
#     """Calculates the CMMD distance between reference and eval image sets.

#     Args:
#       ref_dir: Path to the directory containing reference images.
#       eval_dir: Path to the directory containing images to be evaluated.
#       ref_embed_file: Path to the pre-computed embedding file for the reference images.
#       batch_size: Batch size used in the CLIP embedding calculation.
#       max_count: Maximum number of images to use from each directory. A
#         non-positive value reads all images available except for the images
#         dropped due to batching.

#     Returns:
#       The CMMD value between the image sets.
#     """
#     if ref_dir and ref_embed_file:
#         raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
#     embedding_model = embedding.ClipEmbeddingModel()
#     if ref_embed_file is not None:
#         ref_embs = np.load(ref_embed_file).astype("float32")
#     else:
#         ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
#             "float32"
#         )
#     eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
#     print(type(eval_embs)) #  <class 'numpy.ndarray'>
#     print(eval_embs.shape) #  (50, 768)
   
       
#     val = distance.mmd(ref_embs, eval_embs)
#     return val.numpy()


def main(argv):
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    _, dir1, dir2 = argv
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
    )


if __name__ == "__main__":
    app.run(main)
