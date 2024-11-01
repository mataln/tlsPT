# Script to generate dummy npy data for development work at office

# Dummy shapes
# Line
# Plane
# Cube
# Quadratic surface
# Sphere

from __future__ import annotations

# Add small gaussian noise to all
import argparse
import os

import numpy as np
import pandas as pd


def random_rotation_matrix(dim):
    """Generates a random rotation matrix in given dimensions."""
    A = np.random.normal(0, 1, (dim, dim))
    Q, R = np.linalg.qr(A)
    # Ensure the rotation matrix has determinant +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def generate_line(n, dim, noise):  # Specified in polar coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    r = np.random.uniform(-1, 1, n)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    data = np.stack([x, y, z], axis=1)

    if noise > 0:
        data += np.random.normal(0, noise, data.shape)

    return data


def generate_plane(
    n, dim, noise
):  # Use the xy plane then rotate about x axis,z axis and translate
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.zeros(n)

    data = np.stack([x, y, z], axis=1)

    if noise > 0:
        data += np.random.normal(0, noise, data.shape)

    # Rotate phi about x axis
    x_rotate = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    z_rotate = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    data = data @ x_rotate
    data = data @ z_rotate

    return data


def generate_quadric_surface(n, dim, noise, scale_range):
    """
    Generates data points on a quadric surface in specified dimensions.
    """
    in_data = np.random.uniform(-1, 1, (n, dim - 1))

    A = np.random.uniform(-1, 1, (dim - 1, dim - 1))
    A = (A + A.T) / 2  # Make it symmetric

    b = np.random.uniform(-1, 1, dim - 1)

    c = np.random.uniform(-1, 1)

    z_quadratic = np.einsum("ij,ij->i", in_data @ A, in_data)
    z_linear = in_data @ b
    z = z_quadratic + z_linear + c

    out_data = np.hstack([in_data, z[:, np.newaxis]])

    scale = np.random.uniform(*scale_range)
    out_data *= scale

    if noise > 0:
        out_data += np.random.normal(0, noise, out_data.shape)

    rotation_matrix = random_rotation_matrix(dim)
    out_data = out_data @ rotation_matrix

    return out_data


def generate_cube(n, dim, noise, scale_range):
    """
    Generates data points on the surface of a cube in specified dimensions.
    Uses rejection sampling to ensure all points are within [-1, 1].
    May clip part of the cube depending on scaling.
    """
    # Set cube scaling, center, and rotation
    scale = np.random.uniform(scale_range[0], scale_range[1])
    center = np.random.uniform(-0.5, 0.5, dim)
    rotation = random_rotation_matrix(dim)

    data = np.empty((0, dim))
    while data.shape[0] < n:
        batch_size = n - data.shape[0]

        num_faces = 2 * dim
        face = np.random.randint(0, num_faces, batch_size)
        axis = face // 2
        sign = face % 2

        new_data = np.random.uniform(-1, 1, (batch_size, dim))

        new_data[np.arange(batch_size), axis] = np.where(sign == 0, 1, -1)

        new_data *= scale

        new_data = new_data @ rotation

        new_data += center

        if noise > 0:
            new_data += np.random.normal(0, noise, (batch_size, dim))

        # Keep points within [-1, 1] in all coordinates
        mask = np.all((new_data >= -1) & (new_data <= 1), axis=1)
        new_data = new_data[mask]

        data = np.vstack([data, new_data])

    # Trim to n points
    data = data[:n]
    return data


def generate_sphere(n, dim, noise=0.0, scale_range=(1.0, 1.0)):
    """
    Generates data points on the surface of a sphere in specified dimensions.

    Parameters:
        n (int): Number of data points to generate.
        dim (int): Dimensionality of the space.
        noise (float): Standard deviation of Gaussian noise to add to the data.
        scale_range (tuple): Range (min, max) for the sphere's radius scaling.

    Returns:
        data (np.ndarray): Array of shape (n, dim) containing the generated data points.
    """
    # Generate random points on a unit sphere using the normal distribution
    data = np.random.normal(0, 1, (n, dim))
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms  # Normalize to lie on the unit sphere

    # Apply random scaling (radius)
    scale = np.random.uniform(*scale_range)
    data *= scale  # Scale the sphere radius

    # Apply random rotation
    rotation_matrix = random_rotation_matrix(dim)
    data = data @ rotation_matrix

    # Apply random translation (center the sphere randomly within [-0.5, 0.5])
    center = np.random.uniform(-0.5, 0.5, dim)
    data += center

    # Add Gaussian noise
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)

    # Use rejection sampling to keep points within [-1, 1] in all dimensions
    mask = np.all((data >= -1) & (data <= 1), axis=1)
    data = data[mask]

    # If not enough points after rejection, repeat until we have n points
    while data.shape[0] < n:
        additional_n = n - data.shape[0]
        additional_data = np.random.normal(
            0, 1, (additional_n * 2, dim)
        )  # Generate extra points
        norms = np.linalg.norm(additional_data, axis=1, keepdims=True)
        additional_data = additional_data / norms
        additional_data *= scale
        additional_data = additional_data @ rotation_matrix
        additional_data += center
        if noise > 0:
            additional_data += np.random.normal(0, noise, additional_data.shape)
        mask = np.all((additional_data >= -1) & (additional_data <= 1), axis=1)
        additional_data = additional_data[mask]
        data = np.vstack([data, additional_data])
    data = data[:n]  # Trim to n points

    return data


implemented_shapes = ["line", "plane", "cube", "quadric_surface", "sphere"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_dummy_data",
        description="Generate dummy data in unit cube for development work",
    )

    parser.add_argument(
        "--out", type=str, default="data/dummy_data/", help="Output directory"
    )
    parser.add_argument(
        "--n", type=int, default=1024, help="Number of samples to generate"
    )
    parser.add_argument("--dim", type=int, default=3, help="Dimension of data")
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Shape of data to generate, defaults to uniform mix of all shapes",
    )
    parser.add_argument(
        "--noise", type=float, default=0.01, help="Noise variance to add to data"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--scalerange",
        nargs=2,
        type=float,
        default=[0.7, 1.3],
        help="Range of scale for shapes",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.6, 0.2, 0.2],
        help="Ratios of train, test and val data",
    )
    parser.add_argument("--v", type=bool, default=False, help="Verbose mode")

    args = parser.parse_args()

    if sum(args.ratios) != 1:
        raise ValueError("Ratios must sum to 1")

    np.random.seed(args.seed)

    if args.shape is not None:
        assert args.shape in implemented_shapes, f"Shape {args.shape} not implemented"
        shapes = [args.shape]
    else:
        shapes = implemented_shapes

    splits = pd.DataFrame(columns=["identifier", "split"])

    generated = 0
    while generated < args.n:
        for shape in shapes:
            if shape == "line":
                data = generate_line(args.n, args.dim, args.noise)
            elif shape == "plane":
                data = generate_plane(args.n, args.dim, args.noise)
            elif shape == "cube":
                data = generate_cube(args.n, args.dim, args.noise, args.scalerange)
            elif shape == "quadric_surface":
                data = generate_quadric_surface(
                    args.n, args.dim, args.noise, args.scalerange
                )
            elif shape == "sphere":
                data = generate_sphere(args.n, args.dim, args.noise, args.scalerange)
            else:
                raise NotImplementedError(f"Shape {shape} not implemented")

            out_path = os.path.join(args.out, "raw")
            save_path = os.path.join(out_path, f"{shape}_{generated}.npy")
            os.makedirs(out_path, exist_ok=True)
            np.save(save_path, data)

            if args.v:
                print(f"Saved {out_path}/{shape}_{generated}.npy")
            splits = pd.concat(
                [
                    splits,
                    pd.DataFrame(
                        {"identifier": [f"{shape}_{generated}"], "split": [None]}
                    ),
                ],
                ignore_index=True,
            )
            generated += 1
            if generated >= args.n:
                break

    # shuffle splits file
    splits = splits.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split data
    train_split = int(args.ratios[0] * args.n)
    test_split = int(args.ratios[1] * args.n)
    val_split = args.n - train_split - test_split

    splits.loc[:train_split, "split"] = "train"
    splits.loc[train_split : train_split + test_split, "split"] = "test"
    splits.loc[train_split + test_split :, "split"] = "val"

    # Shuffle again
    splits = splits.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    splits.to_csv(os.path.join(args.out, "splits.csv"), index=False)

    print(f"Generated {generated} samples in total")
