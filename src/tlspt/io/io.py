from __future__ import annotations

import numpy as np

from tlspt import utils

from tlspt.structures.pointclouds import TLSPointclouds
from pytorch3d.io.pluggable_formats import PointcloudFormatInterpreter, endswith 
from collections import namedtuple
from iopath.common.file_io import PathManager
from dataclasses import dataclass
from typing import Optional, List

from pytorch3d.io.utils import PathOrStr, _open_file, _make_tensor
from pytorch3d.io.ply_io import (
    _load_ply_raw, 
    _write_ply_header, 
    _check_faces_indices, 
    _PlyHeader, 
    _PlyElementType,)

import torch
import sys

from dataclasses import asdict

from io import BytesIO

from loguru import logger

_PlyTypeData = namedtuple("_PlyTypeData", "size struct_char np_type")
_PLY_TYPES = {
    "char": _PlyTypeData(1, "b", np.byte),
    "uchar": _PlyTypeData(1, "B", np.ubyte),
    "short": _PlyTypeData(2, "h", np.short),
    "ushort": _PlyTypeData(2, "H", np.ushort),
    "int": _PlyTypeData(4, "i", np.int32),
    "uint": _PlyTypeData(4, "I", np.uint32),
    "float": _PlyTypeData(4, "f", np.float32),
    "double": _PlyTypeData(8, "d", np.float64),
    "int8": _PlyTypeData(1, "b", np.byte),
    "uint8": _PlyTypeData(1, "B", np.ubyte),
    "int16": _PlyTypeData(2, "h", np.short),
    "uint16": _PlyTypeData(2, "H", np.ushort),
    "int32": _PlyTypeData(4, "i", np.int32),
    "uint32": _PlyTypeData(4, "I", np.uint32),
    "float32": _PlyTypeData(4, "f", np.float32),
    "float64": _PlyTypeData(8, "d", np.float64),
}

@dataclass(frozen=True)
class _VertsColumnIndices:
    """
    Contains the relevant layout of the verts section of file being read.
    Members
        point_idxs: List[int] of 3 point columns.
        color_idxs: List[int] of 3 color columns if they are present,
                    otherwise None.
        color_scale: value to scale colors by.
        normal_idxs: List[int] of 3 normals columns if they are present,
                    otherwise None.
    """

    point_idxs: List[int]
    feature_idxs: Optional[List[int]]
    feature_names: Optional[List[str]]
    color_scale: float
    normal_idxs: Optional[List[int]]
    texture_uv_idxs: Optional[List[int]]

@dataclass(frozen=True)
class _VertsData:
    """
    Contains the data of the verts section of file being read.
    Members:
        verts: FloatTensor of shape (V, 3).
        verts_colors: None or FloatTensor of shape (V, 3).
        verts_normals: None or FloatTensor of shape (V, 3).
    """

    verts: torch.Tensor
    verts_features: Optional[torch.Tensor] = None
    feature_names: Optional[List[str]] = None
    verts_normals: Optional[torch.Tensor] = None
    verts_texture_uvs: Optional[torch.Tensor] = None

@dataclass(frozen=True)
class _PlyData:
    """
    Contains the data from a PLY file which has been read.
    Members:
        header: _PlyHeader of file metadata from the header
        verts: FloatTensor of shape (V, 3).
        faces: None or LongTensor of vertex indices, shape (F, 3).
        verts_colors: None or FloatTensor of shape (V, 3).
        verts_normals: None or FloatTensor of shape (V, 3).
    """

    header: _PlyHeader
    verts: torch.Tensor
    faces: Optional[torch.Tensor]
    verts_features: Optional[torch.Tensor]
    feature_names: Optional[List[str]]
    verts_normals: Optional[torch.Tensor]
    verts_texture_uvs: Optional[torch.Tensor]

def _get_verts_column_indices(
    vertex_head: _PlyElementType,
) -> _VertsColumnIndices:
    """
    Get the columns of verts, verts_colors, and verts_normals in the vertex
    element of a parsed ply file, together with a color scale factor.
    When the colors are in byte format, they are scaled from 0..255 to [0,1].
    Otherwise they are not scaled.

    For example, if the vertex element looks as follows:

        element vertex 892
        property double x
        property double y
        property double z
        property double nx
        property double ny
        property double nz
        property uchar red
        property uchar green
        property uchar blue
        property double texture_u
        property double texture_v

    then the return value will be ([0,1,2], [6,7,8], 1.0/255, [3,4,5])

    Args:
        vertex_head: as returned from load_ply_raw.

    Returns:
        _VertsColumnIndices object
    """
    point_idxs: List[Optional[int]] = [None, None, None]
    color_idxs: List[Optional[int]] = [None, None, None]
    normal_idxs: List[Optional[int]] = [None, None, None]
    feature_idxs: List[Optional[int]] = []
    feature_names: List[Optional[str]] = []
    texture_uv_idxs: List[Optional[int]] = [None, None]
    for i, prop in enumerate(vertex_head.properties):
        stashed = False
        if prop.list_size_type is not None:
            raise ValueError("Invalid vertices in file: did not expect list.")
        for j, letter in enumerate(["x", "y", "z"]):
            if prop.name == letter:
                point_idxs[j] = i
                stashed = True
        for j, name in enumerate(["red", "green", "blue"]):
            if prop.name == name:
                color_idxs[j] = i
                stashed = True
        for j, name in enumerate(["nx", "ny", "nz"]):
            if prop.name == name:
                normal_idxs[j] = i
                stashed = True
        for j, name in enumerate(["texture_u", "texture_v"]):
            if prop.name == name:
                texture_uv_idxs[j] = i
                stashed = True
        if not stashed:
            feature_idxs.append(i) #There is a feature at index i
            feature_names.append(prop.name)

    feature_idxs = feature_idxs if None in color_idxs else color_idxs + feature_idxs
    feature_names = feature_names if None in color_idxs else ["red", "green", "blue"] + feature_names



    if None in point_idxs:
        raise ValueError("Invalid vertices in file.")
    color_scale = 1.0
    if all(
        idx is not None and _PLY_TYPES[vertex_head.properties[idx].data_type].size == 1
        for idx in color_idxs
    ):
        color_scale = 1.0 / 255
    return _VertsColumnIndices(
        point_idxs=point_idxs,
        feature_idxs=feature_idxs if feature_idxs else None,
        feature_names=feature_names if feature_idxs else None,
        color_scale=color_scale,
        normal_idxs=None if None in normal_idxs else normal_idxs,
        texture_uv_idxs=None if None in texture_uv_idxs else texture_uv_idxs,
    )

def _get_verts(header: _PlyHeader, elements: dict) -> _VertsData:
    """
    Get the vertex locations, colors and normals from a parsed ply file.

    Args:
        header, elements: as returned from load_ply_raw.

    Returns:
        _VertsData object
    """

    vertex = elements.get("vertex", None)
    if vertex is None:
        raise ValueError("The ply file has no vertex element.")
    if not isinstance(vertex, list):
        raise ValueError("Invalid vertices in file.")
    vertex_head = next(head for head in header.elements if head.name == "vertex")

    column_idxs = _get_verts_column_indices(vertex_head)

    # Case of no vertices
    if vertex_head.count == 0:
        verts = torch.zeros((0, 3), dtype=torch.float32)
        if column_idxs.feature_idxs is None:
            return _VertsData(verts=verts)
        return _VertsData(
            verts=verts, verts_features=torch.zeros((0, len(column_idxs.feature_names)), dtype=torch.float32)
        )

    # Simple case where the only data is the vertices themselves
    if (
        len(vertex) == 1
        and isinstance(vertex[0], np.ndarray)
        and vertex[0].ndim == 2
        and vertex[0].shape[1] == 3
    ):
        return _VertsData(verts=_make_tensor(vertex[0], cols=3, dtype=torch.float32))

    vertex_features = None
    vertex_normals = None
    vertex_texture_uvs = None

    if len(vertex) == 1:
        # This is the case where the whole vertex element has one type,
        # so it was read as a single array and we can index straight into it.
        verts = torch.tensor(vertex[0][:, column_idxs.point_idxs], dtype=torch.float32)
        if column_idxs.feature_idxs is not None:
            vertex_features = torch.tensor(
                vertex[0][:, column_idxs.feature_idxs], dtype=torch.float32
            )
            #Color scale the features corresponding to colors
            color_idxs = [i for i, name in enumerate(column_idxs.feature_names) if name in ["red", "green", "blue"]]
            if color_idxs:
                vertex_features[:, color_idxs] *= column_idxs.color_scale
        if column_idxs.normal_idxs is not None:
            vertex_normals = torch.tensor(
                vertex[0][:, column_idxs.normal_idxs], dtype=torch.float32
            )
        if column_idxs.texture_uv_idxs is not None:
            vertex_texture_uvs = torch.tensor(
                vertex[0][:, column_idxs.texture_uv_idxs], dtype=torch.float32
            )
    else:
        # The vertex element is heterogeneous. It was read as several arrays,
        # part by part, where a part is a set of properties with the same type.
        # For each property (=column in the file), we store in
        # prop_to_partnum_col its partnum (i.e. the index of what part it is
        # in) and its column number (its index within its part).
        prop_to_partnum_col = [
            (partnum, col)
            for partnum, array in enumerate(vertex)
            for col in range(array.shape[1])
        ]
        verts = torch.empty(size=(vertex_head.count, 3), dtype=torch.float32)
        for axis in range(3):
            partnum, col = prop_to_partnum_col[column_idxs.point_idxs[axis]]
            verts.numpy()[:, axis] = vertex[partnum][:, col]
            # Note that in the previous line, we made the assignment
            # as numpy arrays by casting verts. If we took the (more
            # obvious) method of converting the right hand side to
            # torch, then we might have an extra data copy because
            # torch wants contiguity. The code would be like:
            #   if not vertex[partnum].flags["C_CONTIGUOUS"]:
            #      vertex[partnum] = np.ascontiguousarray(vertex[partnum])
            #   verts[:, axis] = torch.tensor((vertex[partnum][:, col]))
        if column_idxs.feature_idxs is not None:
            vertex_features = torch.empty(
                size=(vertex_head.count, len(column_idxs.feature_idxs)), dtype=torch.float32
            )
            for i, feature_idx in enumerate(column_idxs.feature_idxs):
                partnum, col = prop_to_partnum_col[feature_idx]
                vertex_features.numpy()[:, i] = vertex[partnum][:, col]
            color_idxs = [i for i, name in enumerate(column_idxs.feature_names) if name in ["red", "green", "blue"]]
            if color_idxs:
                vertex_features[:, color_idxs] *= column_idxs.color_scale
        if column_idxs.normal_idxs is not None:
            vertex_normals = torch.empty(
                size=(vertex_head.count, 3), dtype=torch.float32
            )
            for axis in range(3):
                partnum, col = prop_to_partnum_col[column_idxs.normal_idxs[axis]]
                vertex_normals.numpy()[:, axis] = vertex[partnum][:, col]
        if column_idxs.texture_uv_idxs is not None:
            vertex_texture_uvs = torch.empty(
                size=(vertex_head.count, 2),
                dtype=torch.float32,
            )
            for axis in range(2):
                partnum, col = prop_to_partnum_col[column_idxs.texture_uv_idxs[axis]]
                vertex_texture_uvs.numpy()[:, axis] = vertex[partnum][:, col]
    return _VertsData(
        verts=verts,
        verts_features=vertex_features,
        feature_names=column_idxs.feature_names,
        verts_normals=vertex_normals,
        verts_texture_uvs=vertex_texture_uvs,
    )

def load_numpy(file_path: str, out_dtype: np.dtype = np.float32, allow_pickle: bool = False) -> np.ndarray:
    """
    Loads a numpy file from disk
    """
    if not file_path.endswith(".npy"):
        raise ValueError(f"file {file_path} is not a numpy file")

    if not utils.check_file_exists(file_path):
        raise ValueError(f"cannot find file at {file_path}")

    arr = np.load(file_path, allow_pickle=allow_pickle)
    arr = arr.astype(out_dtype)
    return arr

def _load_ply(f, *, path_manager: PathManager) -> _PlyData:
    """
    Load the data from a .ply file.

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.
        path_manager: PathManager for loading if f is a str.

    Returns:
        _PlyData object
    """
    header, elements = _load_ply_raw(f, path_manager=path_manager)
    verts_data = _get_verts(header, elements)

    #vertex_head = next(head for head in header.elements if head.name == "vertex")
    #print(vertex_head.properties)


    face = elements.get("face", None)
    if face is not None:
        face_head = next(head for head in header.elements if head.name == "face")
        if (
            len(face_head.properties) != 1
            or face_head.properties[0].list_size_type is None
        ):
            raise ValueError("Unexpected form of faces data.")
        # face_head.properties[0].name is usually "vertex_index" or "vertex_indices"
        # but we don't need to enforce this.

    if face is None:
        faces = None
    elif not len(face):
        # pyre is happier when this condition is not joined to the
        # previous one with `or`.
        faces = None
    elif isinstance(face, np.ndarray) and face.ndim == 2:  # Homogeneous elements
        if face.shape[1] < 3:
            raise ValueError("Faces must have at least 3 vertices.")
        face_arrays = [face[:, [0, i + 1, i + 2]] for i in range(face.shape[1] - 2)]
        faces = torch.LongTensor(np.vstack(face_arrays).astype(np.int64))
    else:
        face_list = []
        for (face_item,) in face:
            if face_item.ndim != 1:
                raise ValueError("Bad face data.")
            if face_item.shape[0] < 3:
                raise ValueError("Faces must have at least 3 vertices.")
            for i in range(face_item.shape[0] - 2):
                face_list.append([face_item[0], face_item[i + 1], face_item[i + 2]])
        faces = torch.tensor(face_list, dtype=torch.int64)

    if faces is not None:
        _check_faces_indices(faces, max_index=verts_data.verts.shape[0])

    return _PlyData(**asdict(verts_data), faces=faces, header=header)

def _write_ply_header(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor],
    verts_features: Optional[torch.Tensor],
    feature_names: Optional[List[str]],
    ascii: bool,
    colors_as_uint8: bool,
) -> None:
    """
    Internal implementation for writing header when saving to a .ply file.

    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_features: FloatTensor of shape (V, C) giving vertex features, including colors.
        ascii: (bool) whether to use the ascii ply format.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert faces is None or not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)
    assert verts_normals is None or (
        verts_normals.dim() == 2 and verts_normals.size(1) == 3
    )
    assert verts_features is None or (verts_features.dim() == 2)

    if ascii:
        f.write(b"ply\nformat ascii 1.0\n")
    elif sys.byteorder == "big":
        f.write(b"ply\nformat binary_big_endian 1.0\n")
    else:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {verts.shape[0]}\n".encode("ascii"))
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    if verts_normals is not None:
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
    if verts_features is not None:
        color_ply_type = b"uchar" if colors_as_uint8 else b"float"
        for feature_name in feature_names:
            if feature_name in ["red", "green", "blue"]:
                f.write(b"property " + color_ply_type + b" " + feature_name.encode("ascii") + b"\n")
            else:
                f.write(b"property float " + feature_name.encode("ascii") + b"\n")
    if len(verts) and faces is not None:
        f.write(f"element face {faces.shape[0]}\n".encode("ascii"))
        f.write(b"property list uchar int vertex_index\n")
    f.write(b"end_header\n")

def _save_ply(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor],
    verts_features: Optional[torch.Tensor],
    feature_names: Optional[List[str]],
    ascii: bool,
    decimal_places: Optional[int] = None,
    colors_as_uint8: bool,
) -> None:
    """
    Internal implementation for saving 3D data to a .ply file.

    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors.
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    _write_ply_header(
        f,
        verts=verts,
        faces=faces,
        verts_normals=verts_normals,
        verts_features=verts_features,
        feature_names=feature_names,
        ascii=ascii,
        colors_as_uint8=colors_as_uint8,
    )

    if not (len(verts)):
        logger.warning("Empty 'verts' provided")
        return

    color_np_type = np.ubyte if colors_as_uint8 else np.float32
    verts_dtype = [("verts", np.float32, 3)]
    if verts_normals is not None:
        verts_dtype.append(("normals", np.float32, 3))
    if verts_features is not None:
        # pyre-fixme[6]: For 1st argument expected `Tuple[str,
        #  Type[floating[_32Bit]], int]` but got `Tuple[str,
        #  Type[Union[floating[_32Bit], unsignedinteger[typing.Any]]], int]`.
        verts_dtype.append(("colors", color_np_type, 3))
        verts_dtype.append(("features", np.float32, verts_features.shape[1]))

    vert_data = np.zeros(verts.shape[0], dtype=verts_dtype)
    vert_data["verts"] = verts.detach().cpu().numpy()
    if verts_normals is not None:
        vert_data["normals"] = verts_normals.detach().cpu().numpy()
    if verts_features is not None:
        feature_data = verts_features.detach().cpu().numpy()
        color_idxs = [i for i, name in enumerate(feature_names) if name in ["red", "green", "blue"]]
        vert_data["features"] = feature_data
        if colors_as_uint8:
            vert_data["featurse"][:, color_idxs] = np.rint(feature_data[:, color_idxs] * 255)
            #vert_data["colors"] = np.rint(color_data * 255)

    if ascii:
        if decimal_places is None:
            float_str = b"%f"
        else:
            float_str = b"%" + b".%df" % decimal_places
        float_group_str = (float_str + b" ") * 3
        formats = [float_group_str]
        if verts_normals is not None:
            formats.append(float_group_str)
        if verts_features is not None:
            formats.append(b"%d %d %d " if colors_as_uint8 else float_group_str)
        formats[-1] = formats[-1][:-1] + b"\n"
        for line_data in vert_data:
            for data, format in zip(line_data, formats):
                f.write(format % tuple(data))
    else:
        if isinstance(f, BytesIO):
            # tofile only works with real files, but is faster than this.
            f.write(vert_data.tobytes())
        else:
            vert_data.tofile(f)

    if faces is not None:
        faces_array = faces.detach().cpu().numpy()

        _check_faces_indices(faces, max_index=verts.shape[0])

        if len(faces_array):
            if ascii:
                np.savetxt(f, faces_array, "3 %d %d %d")
            else:
                faces_recs = np.zeros(
                    len(faces_array),
                    dtype=[("count", np.uint8), ("vertex_indices", np.uint32, 3)],
                )
                faces_recs["count"] = 3
                faces_recs["vertex_indices"] = faces_array
                faces_uints = faces_recs.view(np.uint8)

                if isinstance(f, BytesIO):
                    f.write(faces_uints.tobytes())
                else:
                    faces_uints.tofile(f)

class PointcloudPlyFormat(PointcloudFormatInterpreter):
    def __init__(self) -> None:
        self.known_suffixes = (".ply",)

    def read(
        self,
        path: PathOrStr,
        device,
        path_manager: PathManager,
        **kwargs,
    ) -> Optional[TLSPointclouds]:
        if not endswith(path, self.known_suffixes):
            return None
        data = _load_ply(f=path, path_manager=path_manager)
        features = None
        if data.verts_features is not None:
            features = [data.verts_features.to(device)]
        normals = None
        if data.verts_normals is not None:
            normals = [data.verts_normals.to(device)]

        pointcloud = TLSPointclouds(
            points=[data.verts.to(device)], 
            features=features,
            feature_names=data.feature_names,
            normals=normals
        )
        return pointcloud

    def save(
        self,
        data: TLSPointclouds,
        path: PathOrStr,
        path_manager: PathManager,
        binary: Optional[bool],
        decimal_places: Optional[int] = None,
        colors_as_uint8: bool = False,
        **kwargs,
    ) -> bool:
        """
        Extra optional args:
            colors_as_uint8: (bool) Whether to save colors as numbers in the
                        range [0, 255] instead of float32.
        """
        if not endswith(path, self.known_suffixes):
            return False

        points = data.points_list()[0]
        features = data.features_packed()
        normals = data.normals_packed()
        feature_names = data._feature_names

        with _open_file(path, path_manager, "wb") as f:
            _save_ply(
                f=f,
                verts=points,
                verts_features=features,
                feature_names=feature_names,
                verts_normals=normals,
                faces=None,
                ascii=binary is False,
                decimal_places=decimal_places,
                colors_as_uint8=colors_as_uint8,
            )
        return True