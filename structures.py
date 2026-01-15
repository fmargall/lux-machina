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
  -1: Bounding box. No requirements.
   0: Ideal lens. Requires:
    - v0: wp.vec2 (Point 1)
    - v1: wp.vec2 (Point 2)
    - f0: wp.float32 (focal length)
   1: Segment. Requires:
    - v0: wp.vec2 (Origin) # From v0 and v1 the normal is computed
    - v1: wp.vec2 (End)    # (ie. +90° in trigonometric direction)
    - f0: wp.float32   (refractive index in the +normal direction)
    - f1: wp.float32   (refractive index in the -normal direction)
   2: Circular arc. Requires:
    - v0: wp.vec2    (Origin) # Center of circle
    - f0: wp.float32 (Radius)
    - f1: wp.float32 (Angle of starting point)
    - f2: wp.float32 (Angle of ending point)
    - f3: wp.float32 (refractive index in the +normal direction)
    - f4: wp.float32 (refractive index of the -normal direction)
   3: Aspheric lens. Requires:
    - v0 : wp.vec2    (Point 1) # Equivalent to x = -1 point
    - v1 : wp.vec2    (Point 2) # Equivalent to x =  1 point
    - f0 : wp.float32 (Radius)
    - f1 : wp.float32 (Conic constant)
    - f2 : wp.float32 (refractive index in the +normal direction)
    - f3 : wp.float32 (refractive index in the -normal direction)
    - f4 : wp.float32 (y-intercept) 
    - f5 : wp.float32 (normalization factor)
    - f6 : wp.float32 (1st coefficient, associated to  2nd power)
    - f7 : wp.float32 (2nd coefficient, associated to  4th power)
    - f8 : wp.float32 (3rd coefficient, associated to  6th power)
    - f9 : wp.float32 (4th coefficient, associated to  8th power)
    - f10: wp.float32 (5th coefficient, associated to 10th power)
"""
@wp.struct
class Primitive:
    type: wp.int32
    v0  : wp.vec2
    v1  : wp.vec2
    f0  : wp.float32
    f1  : wp.float32
    f2  : wp.float32
    f3  : wp.float32
    f4  : wp.float32
    f5  : wp.float32
    f6  : wp.float32
    f7  : wp.float32
    f8  : wp.float32
    f9  : wp.float32
    f10 : wp.float32

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