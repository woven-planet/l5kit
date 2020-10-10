from .box_rasterizer import BoxRasterizer
from .map_and_box_rasterizer import MapAndBoxRasterizer
from .rasterizer import Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import SemanticRasterizer


class SemBoxRasterizer(MapAndBoxRasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class"""

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        box_rast: BoxRasterizer,
        sem_rast: SemanticRasterizer,
    ):
        super().__init__(
            render_context=render_context,
            filter_agents_threshold=filter_agents_threshold,
            history_num_frames=history_num_frames,
        )
        self._box_rast = box_rast
        self._sem_rast = sem_rast

    @property
    def box_rast(self) -> Rasterizer:
        return self._box_rast

    @property
    def map_rast(self) -> Rasterizer:
        return self._sem_rast
