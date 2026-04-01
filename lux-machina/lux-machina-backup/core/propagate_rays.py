import warp as wp

from .structures import _Intersection3D, _Ray3D

@wp.kernel
def _propagateRays(
    intersectionsBuffer: wp.array(dtype=_Intersection3D, ndim=1),
    seed               : wp.int32,

    raysBuffer: wp.array(dtype=_Ray3D, ndim=1)
):
    ID = wp.tid()
    
    seed = (seed * raysBuffer.shape[0] + ID) * 1664525 + 1013904223
    rand = wp.randf(wp.uint32(seed))

    intersection = intersectionsBuffer[ID]
    ray          = intersection.ray

    if intersection.hit:
        pass

    else:
        raysBuffer[ID] = _Ray3D(
            isAlive    = False,
            origin     = ray.origin,
            direction  = ray.direction,
            depth      = ray.depth,
            energy     = ray.energy,
            wavelength = ray.wavelength,
            ioState    = ray.ioState
        )