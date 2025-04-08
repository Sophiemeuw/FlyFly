from .arenas import (
    OdorTargetOnlyArena,
    ScatteredPillarsArena,
    LoomingBallArena,
    HierarchicalArena,
    FoodToNestArena,
)

__all__ = ["BaseController"]
levels = {
    0: OdorTargetOnlyArena,
    1: ScatteredPillarsArena,
    2: LoomingBallArena,
    3: HierarchicalArena,
    4: FoodToNestArena,
}
