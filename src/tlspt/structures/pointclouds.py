from __future__ import annotations

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
        if isinstance(feature_names, list):
            if len(feature_names) != self.features_list()[0].shape[1]:
                raise ValueError(
                    "Number of feature names must match number of feature channels"
                )
            if not (all(isinstance(x, str) for x in feature_names)):
                raise ValueError("Feature names must be strings")
        elif feature_names is not None:
            raise ValueError("Feature indices must be a list")
        self._feature_names = feature_names
