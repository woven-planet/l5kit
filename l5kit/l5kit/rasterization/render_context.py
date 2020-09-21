import numpy as np


class RenderContext:
    def __init__(
        self, raster_size_px: np.ndarray, pixel_size_m: np.ndarray, center_in_raster_ratio: np.ndarray,
    ) -> None:
        """
        Arguments:
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

        scaling = 1.0 / pixel_size_m  # scaling factor from world to raster space [pixels per meter]
        center_in_raster_px = center_in_raster_ratio * raster_size_px
        self.raster_from_context = np.array([[scaling[0], 0, center_in_raster_px[0]],
                                             [0, scaling[1], center_in_raster_px[1]],
                                             [0, 0, 1]])

    def raster_from_context(self) -> np.ndarray:
        """
        Return the transform to raster pixels from the render context pose in meters.
        """
        return self.raster_from_context
