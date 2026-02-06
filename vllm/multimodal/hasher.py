# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import hashlib
import pickle
import uuid
from collections.abc import Callable, Iterable

import numpy as np
import torch
from PIL import Image

import vllm.envs as envs
from vllm.logger import init_logger

from .media import MediaWithBytes

logger = init_logger(__name__)


@functools.lru_cache(maxsize=3)
def _get_hasher_factory(algorithm: str) -> Callable[[], "hashlib._Hash"]:
    """
    Get the hasher factory based on the configured algorithm.

    Args:
        algorithm: Hash algorithm name (blake3, sha256, or sha512)

    Returns a callable that creates a new hasher instance.
    Supports blake3 (default), sha256, and sha512 for FIPS compliance.

    See: https://github.com/vllm-project/vllm/issues/18334
    """
    algorithm = algorithm.lower()

    if algorithm == "blake3":
        from blake3 import blake3

        return blake3
    elif algorithm == "sha256":
        return hashlib.sha256
    elif algorithm == "sha512":
        return hashlib.sha512
    else:
        # This should never happen due to env_with_choices validation
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


class MultiModalHasher:
    @classmethod
    def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]:
        # Simple cases
        if isinstance(obj, (bytes, memoryview)):
            return (obj,)
        if isinstance(obj, str):
            return (obj.encode("utf-8"),)
        if isinstance(obj, (int, float)):
            return (np.array(obj).tobytes(),)

        if isinstance(obj, Image.Image):
            exif = obj.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)

            data = {"mode": obj.mode, "data": np.asarray(obj)}
            palette = obj.palette
            if palette is not None:
                data["palette"] = palette.palette
                if palette.rawmode is not None:
                    data["palette_rawmode"] = palette.rawmode

            return cls.iter_item_to_bytes("image", data)

        if isinstance(obj, MediaWithBytes) and isinstance(obj.media, Image.Image):
            exif = obj.media.getexif()
            if Image.ExifTags.Base.ImageID in exif and isinstance(
                exif[Image.ExifTags.Base.ImageID], uuid.UUID
            ):
                return (exif[Image.ExifTags.Base.ImageID].bytes,)

            return cls.iter_item_to_bytes("image", obj.original_bytes)

        if isinstance(obj, torch.Tensor):
            tensor_obj: torch.Tensor = obj.cpu()
            tensor_dtype = tensor_obj.dtype
            tensor_shape = tensor_obj.shape

            # NumPy does not support bfloat16.
            # Workaround: View the tensor as a contiguous 1D array of bytes
            if tensor_dtype == torch.bfloat16:
                tensor_obj = tensor_obj.contiguous()
                tensor_obj = tensor_obj.view((tensor_obj.numel(),)).view(torch.uint8)

                return cls.iter_item_to_bytes(
                    "tensor",
                    {
                        "original_dtype": str(tensor_dtype),
                        "original_shape": tuple(tensor_shape),
                        "data": tensor_obj.numpy(),
                    },
                )
            return cls.iter_item_to_bytes("tensor", tensor_obj.numpy())
        if isinstance(obj, np.ndarray):
            # If the array is non-contiguous, we need to copy it first
            arr_data = (
                obj.view(np.uint8).data if obj.flags.c_contiguous else obj.tobytes()
            )
            return cls.iter_item_to_bytes(
                "ndarray",
                {
                    "dtype": obj.dtype.str,
                    "shape": obj.shape,
                    "data": arr_data,
                },
            )
        logger.warning(
            "No serialization method found for %s. Falling back to pickle.", type(obj)
        )

        return (pickle.dumps(obj),)

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
    ) -> Iterable[bytes | memoryview]:
        if obj is None:
            yield key.encode("utf-8")
            return
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                yield from cls.iter_item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.iter_item_to_bytes(f"{key}.{k}", v)
        else:
            yield key.encode("utf-8")
            yield from cls.serialize_item(obj)

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        hasher_factory = _get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)
        hasher = hasher_factory()

        for k, v in kwargs.items():
            for bytes_ in cls.iter_item_to_bytes(k, v):
                hasher.update(bytes_)

        return hasher.hexdigest()

    @classmethod
    def hash_frame_pair(cls, frame_data: torch.Tensor, pair_index: int) -> str:
        """
        Hash a pair of frames based on pixel content only (position-independent).

        This allows frame-pairs with identical content to match across different
        videos, even if they appear at different positions. This is essential
        for caching overlapping video crops.

        Args:
            frame_data: Tensor containing 2 frames worth of pixel data
            pair_index: Index of this frame-pair in the video (0, 1, 2, ...)
                       Currently unused - kept for API compatibility

        Returns:
            Hex digest hash string
        """
        hasher_factory = _get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)
        hasher = hasher_factory()

        # Hash ONLY pixel data (no position) so identical content matches
        # across videos regardless of position
        # Note: pair_index parameter kept for API compatibility but not used

        # Serialize frame data
        for bytes_ in cls.serialize_item(frame_data):
            hasher.update(bytes_)

        return hasher.hexdigest()

    @classmethod
    def hash_video_frame_pairs(
        cls,
        pixel_values: torch.Tensor,
        temporal_patch_size: int = 2
    ) -> tuple[list[str], str]:
        """
        Generate frame-pair hashes for a video.

        Args:
            pixel_values: Video frames tensor (num_frames, ...)
            temporal_patch_size: Number of frames per pair

        Returns:
            - List of frame-pair hashes (one per pair)
            - Composite video hash (for backward compatibility)
        """
        num_frames = pixel_values.shape[0]
        num_pairs = (num_frames + temporal_patch_size - 1) // temporal_patch_size

        frame_pair_hashes = []
        for i in range(num_pairs):
            start_idx = i * temporal_patch_size
            end_idx = min((i + 1) * temporal_patch_size, num_frames)
            frame_pair = pixel_values[start_idx:end_idx]

            pair_hash = cls.hash_frame_pair(frame_pair, i)
            frame_pair_hashes.append(pair_hash)

        # Composite hash for backward compatibility
        composite_hash = cls.hash_kwargs(
            frame_pair_hashes=frame_pair_hashes,
            num_frames=num_frames
        )

        return frame_pair_hashes, composite_hash
