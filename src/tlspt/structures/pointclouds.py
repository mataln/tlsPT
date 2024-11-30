from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger
from omegaconf import ListConfig
from pytorch3d.structures import Pointclouds


class TLSPointclouds(Pointclouds):
    def __init__(self, points, normals=None, features=None, feature_names=None):
        """
        Args:
            points:
                Can be either

                - List where each element is a tensor of shape (num_points, 3)
                  containing the (x, y, z) coordinates of each point.
                - Padded float tensor with shape (num_clouds, num_points, 3).
            normals:
                Can be either

                - None
                - List where each element is a tensor of shape (num_points, 3)
                  containing the normal vector for each point.
                - Padded float tensor of shape (num_clouds, num_points, 3).
            features:
                Can be either

                - None
                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
                - Padded float tensor of shape (num_clouds, num_points, C).
                where C is the number of channels in the features.
                For example 3 for RGB color.
            feature_names:
                None or
                list of feature names, in order matching the feature channels in the features tensor.

            Note: I have modified the io loading functions to load all
            features, and not just rgb

        Refer to comments above for descriptions of List and Padded
        representations.
        """
        super().__init__(points=points, normals=normals, features=features)
        if isinstance(feature_names, ListConfig):
            feature_names = list(feature_names)
        if isinstance(feature_names, list):  # List
            if len(feature_names) != self.features_list()[0].shape[1]:
                raise ValueError(
                    "Number of feature names must match number of feature channels"
                )
            if not (all(isinstance(x, str) for x in feature_names)):
                raise ValueError("Feature names must be strings")
        elif feature_names is not None:  # Not list or None
            raise ValueError("Feature names must be a list")
        else:  # None
            if features is not None:
                raise ValueError(
                    "Feature names must be provided if features are provided"
                )
        self._feature_names = feature_names


def join_pointclouds_as_batch(
    pointclouds: Sequence[TLSPointclouds], insert_missing_features: bool = False
) -> TLSPointclouds:
    """
    Merge a list of TLSPointclouds objects into a single batched TLSPointclouds
    object. All pointclouds must be on the same device.

    Args:
        batch: List of TLSPointclouds objects each with batch dim [b1, b2, ..., bN]
    Returns:
        pointcloud: Poinclouds object with all input pointclouds collated into
            a single object with batch dim = sum(b1, b2, ..., bN)
    """
    if isinstance(pointclouds, TLSPointclouds) or not isinstance(pointclouds, Sequence):
        raise ValueError("Wrong first argument to join_points_as_batch.")

    device = pointclouds[0].device
    if not all(p.device == device for p in pointclouds):
        raise ValueError("Pointclouds must all be on the same device")

    kwargs = {}

    feature_names_list = [
        getattr(p, "_feature_names") for p in pointclouds
    ]  # The actual features found in each point cloud

    # Use the feature names from the first point cloud as the reference========= n.b this is llm code generated late at night. It might be wrong but I think it's ok.
    reference_feature_names = max(feature_names_list, key=lambda x: len(x))

    # Check if all feature names lists are identical (including order)
    if not all(fn == reference_feature_names for fn in feature_names_list):
        if not insert_missing_features:
            raise ValueError("Point clouds must have the same feature names")
        else:
            # Identify point clouds that need updating
            idxs = [
                i
                for i, fn in enumerate(feature_names_list)
                if fn != reference_feature_names
            ]

            logger.warning(
                f"A total of {len(idxs)} from {len(feature_names_list)} point clouds have missing features. Inserting missing features."
            )

            for i in idxs:
                # Special case: point cloud has no features
                if feature_names_list[i] is None:
                    feature_names_list[i] = reference_feature_names
                    num_points_list = [p.shape[0] for p in pointclouds[i].points_list()]
                    pointclouds[i]._features_list = [
                        torch.full(
                            (n, len(reference_feature_names)),
                            float("nan"),
                            device=device,
                        )
                        for n in num_points_list
                    ]
                    continue  # No need to do anything else

                # Check if all features in this point cloud are in the reference, in the same order
                if feature_names_list[i]:
                    if not all(
                        fn in reference_feature_names for fn in feature_names_list[i]
                    ):
                        raise ValueError(
                            f"Feature mismatch: point cloud {i} has features not present in the reference feature names."
                        )
                    if feature_names_list[i] != [
                        fn
                        for fn in reference_feature_names
                        if fn in feature_names_list[i]
                    ]:
                        raise ValueError(
                            f"Feature order mismatch in point cloud {i}. Feature names must match the reference order."
                        )

                # Insert missing features (present in reference but missing in current point cloud)
                new_features_list = []
                for f in pointclouds[i].features_list():
                    n = f.shape[0]
                    new_f = torch.full(
                        (n, len(reference_feature_names)), float("nan"), device=device
                    )
                    # Copy existing features to the correct positions
                    for idx, fn in enumerate(feature_names_list[i]):
                        ref_idx = reference_feature_names.index(fn)
                        new_f[:, ref_idx] = f[:, idx]
                    new_features_list.append(new_f)
                pointclouds[i]._features_list = new_features_list
                feature_names_list[
                    i
                ] = reference_feature_names  # Update the feature names

            # Final consistency check
            if not all(fn == reference_feature_names for fn in feature_names_list):
                raise ValueError("Failed to insert missing features correctly")
    # ===========================================================================

    kwargs["feature_names"] = feature_names_list[0]

    for field in ("points", "normals", "features"):
        field_list = [getattr(p, field + "_list")() for p in pointclouds]
        if None in field_list:
            if field == "points":
                raise ValueError("Pointclouds cannot have their points set to None!")
            if not all(f is None for f in field_list):
                raise ValueError(
                    f"Pointclouds in the batch have some fields '{field}'"
                    + " defined and some set to None."
                )
            field_list = None
        else:
            field_list = [p for points in field_list for p in points]
            if field == "features" and any(
                p.shape[1] != field_list[0].shape[1] for p in field_list[1:]
            ):
                raise ValueError("Pointclouds must have the same number of features")
        kwargs[field] = field_list

    return TLSPointclouds(**kwargs)


def join_pointclouds_as_scene(
    pointclouds: TLSPointclouds | list[TLSPointclouds],
    insert_missing_features: bool | None = None,
) -> TLSPointclouds:
    """
    Joins a batch of point cloud in the form of a TLSPointclouds object or a list of TLSPointclouds
    objects as a single point cloud. If the input is a list, the TLSPointclouds objects in the
    list must all be on the same device, and they must either all or none have features and
    all or none have normals.

    Args:
        TLSPointclouds: TLSPointclouds object that contains a batch of point clouds, or a list of
                    TLSPointclouds objects.

    Returns:
        new TLSPointclouds object containing a single point cloud
    """
    if isinstance(pointclouds, list):
        if insert_missing_features is None:
            raise ValueError(
                "insert_missing_features must be set if pointclouds is passed as list"
            )
        pointclouds = join_pointclouds_as_batch(pointclouds, insert_missing_features)
    elif insert_missing_features is not None:
        raise ValueError(
            "insert_missing_features can only be set if pointclouds is passed as list"
        )

    if len(pointclouds) == 1:
        return pointclouds
    points = pointclouds.points_packed()
    features = pointclouds.features_packed()
    normals = pointclouds.normals_packed()
    feature_names = pointclouds._feature_names
    pointcloud = TLSPointclouds(
        points=points[None],
        features=None if features is None else features[None],
        normals=None if normals is None else normals[None],
        feature_names=feature_names,
    )
    return pointcloud
