"""
Modified pytorch3d.io.IO to support multiple features
"""
from __future__ import annotations

from pathlib import Path

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io import IO as Pytorch3dIO

from tlspt.io.io import PointcloudPlyFormat
from tlspt.structures.pointclouds import TLSPointclouds


class TLSReader(Pytorch3dIO):
    def __init__(
        self,
        include_default_formats: bool = True,
        path_manager: PathManager | None = None,
    ) -> None:
        super().__init__()

    def register_default_formats(self):
        self.register_pointcloud_format(PointcloudPlyFormat())

    def load_pointcloud(
        self, path: str | Path, device: Device = "cpu", **kwargs
    ) -> TLSPointclouds:
        """
        Attempt to load a point cloud from the given file, using a registered format.

        Args:
            path: file to read
            device: Device (as str or torch.device) on which to load the data.

        Returns:
            new Pointclouds object containing one mesh.
        """
        for pointcloud_interpreter in self.pointcloud_interpreters:
            pointcloud = pointcloud_interpreter.read(
                path, path_manager=self.path_manager, device=device, **kwargs
            )
            if pointcloud is not None:
                return pointcloud

        raise ValueError(f"No point cloud interpreter found to read {path}.")

    def save_pointcloud(
        self,
        data: TLSPointclouds,
        path: str | Path,
        binary: bool | None = None,
        **kwargs,
    ) -> None:
        """
        Attempt to save a point cloud to the given file, using a registered format.

        Args:
            data: a 1-element Pointclouds
            path: file to write
            binary: If there is a choice, whether to save in a binary format.
        """
        if not isinstance(data, TLSPointclouds):
            raise ValueError("Pointclouds object expected.")

        if len(data) != 1:
            raise ValueError("Can only save a single point cloud.")

        for pointcloud_interpreter in self.pointcloud_interpreters:
            success = pointcloud_interpreter.save(
                data, path, path_manager=self.path_manager, binary=binary, **kwargs
            )
            if success:
                return

        raise ValueError(f"No point cloud interpreter found to write to {path}.")
