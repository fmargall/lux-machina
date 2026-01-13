import warp as wp

@wp.struct
class Ray:
    isAlive  : wp.bool
    origin   : wp.vec2
    direction: wp.vec2
    depth    : wp.int32
    energy   : wp.float32

"""
   Types:
   ------
   0: Ideal lens. Requires:
    - v0: wp.vec2 (Point 1)
    - v1: wp.vec2 (Point 2)
    - f0: wp.float32 (focal length)
   1: Segment. Requires:
    - v0: wp.vec2 (Origin) # From v0 and v1 the normal is computed
    - v1: wp.vec2 (End)    # (ie. +90° in trigonometric direction)
    - f0: wp.float32   (refractive index in the +normal direction)
    - f1: wp.float32   (refractive index in the -normal direction)
"""
@wp.struct
class Primitive:
    type: wp.int32
    v0  : wp.vec2
    v1  : wp.vec2
    f0  : wp.float32
    f1  : wp.float32

@wp.struct
class Segment:
    v0: wp.vec2
    v1: wp.vec2
    f0: wp.float32

@wp.struct
class Intersection:
    hit      : wp.bool
    hitPoint : wp.vec2
    normal   : wp.vec2 # surface normal at intersection (pointing outside)
    ray      : Ray
    primitive: Primitive


"""
   Types:
   ------
   0: Pointlight. Requires:
    - v0: wp.vec2  (Origin)
    - f0: wp.float (Intensity)
   1: Lambertian. Requires:
    - v0: wp.vec2 (Origin) # From v0 and v1 the normal is computed
    - v1: wp.vec2 (End)    # (ie. +90° in trigonometric direction)
    - f0: wp.float32 (Intensity)
"""
@wp.struct
class LightSource:
    type: wp.int32
    v0  : wp.vec2
    v1  : wp.vec2
    f0  : wp.float32