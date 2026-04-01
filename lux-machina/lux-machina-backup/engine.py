from contextlib import nullcontext

import warp as wp

from core.generate_primary_rays import _generatePrimaryRays
from core.intersect_rays        import _intersectRays
from core.propagate_rays        import _propagateRays
from core.structures            import _Intersection3D, _Ray3D

# ============================================================
# SCENE
# ============================================================

class Scene:
    def __init__(self):
        self.emitters   = []
        self.primitives = []
        self.sensors    = []
        self.cameras    = []

# ============================================================
# CAMERA
# ============================================================

class Camera:
    def __init__(self, model, requires_grad=False):
        self.model = model

        self.buffer = wp.zeros(
            (model.i0, model.i1),
            dtype=wp.float32,
            requires_grad=requires_grad
        )

    def clear(self):
        self.buffer.zero_()

    def accumulate(self, engine):
        """
        À implémenter avec ton kernel caméra
        """
        raise NotImplementedError

# ============================================================
# ENGINE
# ============================================================

class LuxMachinaEngine:
    def __init__(self, maxDepthLevel: int = 10, nbParallelRays: int = 10_000, optimize: bool = False):
        self._emittersList    = []
        self._primitivesList  = []
        self._sensorsList     = []
        self._camerasList     = []

        self._rasterizersList = []

        # Engine parameters
        self._maxDepthLevel  = maxDepthLevel
        self._nbParallelRays = nbParallelRays

        # Optimization parameters
        self._optimize = optimize

        # _frameID is used for the seeds of the Monte Carlo simulation
        # It should not be modified directly and should be incremented
        # using the associated function. It will be left to zero, when
        # doing optimization.
        self._frameID = 0

        # ----------------------------------------------------
        # BUFFERS
        # ----------------------------------------------------

        self._raysBuffer = wp.empty(nbParallelRays, dtype=_Ray3D, requires_grad=self._optimize)
        self._intersectionsBuffer = wp.empty(nbParallelRays, dtype=_Intersection3D, requires_grad=self._optimize)

        if self._optimize:
            self._lossBuffer = None

        else:
            self._lossBuffer = None

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def render(self):
        # Preparation of the context. When the engine is not only in
        # the 'rendering' mode but also in the 'optimize' mode, it's
        # needed to allow the Tape context to compute the gradient.
        if self._optimize:
            tape = wp.Tape()
        else:
            tape = nullcontext()

        with tape:
            # Generate the primary rays
            self._generatePrimaryRays()

            for depthLevel in range(self._maxDepthLevel):
                # Intersect the primitives
                self._intersectPrimitives()

                # Accumulate for every sensor
                for sensor in self._sensorsList:
                    sensor.accumulate(self)

                # Rasterize if required
                for rasterizer in self._rasterizersList:
                    rasterizer.rasterize(self)

                # Accumulate for every camera
                for camera in self._camerasList:
                    camera.accumulate(self)

                # Propagate rays
                self._propagateRays(depthLevel)

            if self._optimize:
                self._computeLoss()

        if self._optimize:
            tape.backward(loss=self._lossBuffer, grads=None)


    # ========================================================
    # KERNEL WRAPPERS
    # ========================================================

    def _generatePrimaryRays(self):
        wp.launch(
            kernel  =  _generatePrimaryRays,
            dim     =  self._nbParallelRays,
            inputs  = [self._emittersWorldBuffer, self._frameID],
            outputs = [self._raysBuffer]
        )

    def _intersectPrimitives(self):
        wp.launch(
            kernel  =  _intersectRays, 
            dim     =  self._nbParallelRays,
            inputs  = [self._raysBuffer, self._primitivesWorldBuffer, len(self._primitivesList)],
            outputs = [self._intersectionsBuffer]
        )

    def _propagateRays(self, depthLevel: int):
        seed = self._frameID * self._maxDepthLevel + depthLevel

        wp.launch(
            kernel  =  _propagateRays,
            dim     =  self._nbParallelRays,
            inputs  = [self._intersectionsBuffer, seed],
            outputs = [self._raysBuffer]
        )

    # ========================================================
    # LOSS
    # ========================================================

    def _computeLoss(self):
        pass

    # ========================================================
    # FRAME MANAGEMENT
    # ========================================================

    def _incrementFrameID(self):
        if not self._optimize:
            self._frameID += 1

if __name__ == "__main__":
    wp.init()

    # -------------------------
    # Scene
    # -------------------------
    scene = Scene()


