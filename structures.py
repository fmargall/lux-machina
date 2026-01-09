import warp as wp

@wp.struct
class Ray:
    origin   : wp.vec2
    direction: wp.vec2
    depth    : wp.int32
    energy   : wp.float32
    weight   : wp.float32
    alive    : wp.bool

@wp.struct
class Segment:
    s0: wp.vec2
    s1: wp.vec2
    weight: wp.float32

@wp.struct
class Intersection:
    hitPoint: wp.vec2
    normal  : wp.vec2 # surface normal at intersection (pointing outside)
    ni      : wp.float32 # refractive index outside (in front of normal)
    no      : wp.float32 # refractive index inside   (behind the normal)

"""
   Type:
    0: Point light source. Requires:
     - p0: wp.vec2 (Position)
     - intensity: wp.float32
    
    1: Lambertian light source. Requires:
     - p0: wp.vec2 (Start position of segment)
     - p1: wp.vec2 (End position of segment)
     - intensity: wp.float32
"""
@wp.struct
class LightSource:
    type: wp.int32
    intensity: wp.float32
    p0: wp.vec2
    p1: wp.vec2

@wp.struct
class Primitive:
    type: wp.int32