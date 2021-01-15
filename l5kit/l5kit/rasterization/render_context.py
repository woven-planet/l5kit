import numpy as np

from ..geometry import vertical_flip


class RenderContext:
    def __init__(
            self,
            raster_size_px: np.ndarray,
            pixel_size_m: np.ndarray,
            center_in_raster_ratio: np.ndarray,
            set_origin_to_bottom: bool,
    ) -> None:
        """
        This class stores render context information (raster size, pixel size, raster center / principle point) and
        it computes a transformation matrix (raster_from_local) to transform a local coordinates into raster ones.
        (0,0) in local will end up in the center of the raster (specified by combining `raster_size_px` and
        `center_in_raster_ratio`).

        Args:
            raster_size_px (Tuple[int, int]): Raster size in pixels
            pixel_size_m (np.ndarray): Size of one pixel in the real world, meter per pixel
            center_in_raster_ratio (np.ndarray): Where to center the local pose in the raster. [0.5,0.5] would be in
                the raster center, [0, 0] is bottom left.
        """

        if pixel_size_m[0] != pixel_size_m[1]:
            raise NotImplementedError("No support for non squared pixels yet")

        self.raster_size_px = raster_size_px
        self.pixel_size_m = pixel_size_m
        self.center_in_raster_ratio = center_in_raster_ratio
        self.set_origin_to_bottom = set_origin_to_bottom

        scaling = 1.0 / pixel_size_m  # scaling factor from world to raster space [pixels per meter]
        center_in_raster_px = center_in_raster_ratio * raster_size_px
        self.raster_from_local = np.array(
            [[scaling[0], 0, center_in_raster_px[0]], [0, scaling[1], center_in_raster_px[1]], [0, 0, 1]]
        )
        if set_origin_to_bottom:
            self.raster_from_local = vertical_flip(self.raster_from_local, self.raster_size_px[1])

    def raster_from_world(self, position_m: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Return a matrix to convert a pose in world coordinates into raster coordinates

        Args:
            render_context (RenderContext): the context for rasterisation
            position_m (np.ndarray): XY position in world coordinates
            angle_rad (float): rotation angle in world coordinates

        Returns:
            (np.ndarray): a transformation matrix from world coordinates to raster coordinates
        """
        # Compute pose from its position and rotation
        pose_in_world = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), position_m[0]],
                [np.sin(angle_rad), np.cos(angle_rad), position_m[1]],
                [0, 0, 1],
            ]
        )

        pose_from_world = np.linalg.inv(pose_in_world)
        raster_from_world = self.raster_from_local @ pose_from_world
        return raster_from_world
