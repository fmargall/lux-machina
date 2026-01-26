import warp as wp

@wp.struct
class Ray:
    isAlive  : wp.bool
    origin   : wp.vec2
    direction: wp.vec2
    depth    : wp.int32
    energy   : wp.float32

@wp.struct
class Ray3D:
    isAlive  : wp.bool
    origin   : wp.vec3
    direction: wp.vec3
    depth    : wp.int32
    energy   : wp.float32

"""
   Types:
   ------
  -1: Bounding box. Requires:
    - f0: wp.float32 (Aspect ratio)
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
   4: Blocker. Requires:
    - v0: wp.vec2 (Point 1)
    - v1: wp.vec2 (Point 2)
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

"""
   Types:
   ------
   0: Annulus blocker. Requires:
    - v0: wp.vec3    (Origin)
    - v1: wp.vec3    (Normal)
    - f0: wp.float32 (Inner radius)
    - f1: wp.float32 (Outer radius)
   1: Cylinder. Requires:
    - v0: wp.vec3    (Center of basis)
    - v1: wp.vec3    (Normal of basis with height as length)
    - f0: wp.float32 (Radius)
    - f1: wp.float32 (Refractive index in the +normal direction, ie. the outside)
    - f2: wp.float32 (Refractive index in the -normal direction, ie. the inside)
   2: Spherical cap. Requires:
    - v0: wp.vec3    (Center of sphere)
    - v1: wp.vec3    (Direction of spherical cap pole, with radius as length)
    - f0: wp.float32 (Angle of the spherical cap: pi(/2) for an (hemi)sphere)
    - f1: wp.float32 (Refractive index in the +normal direction, ie. the outside)
    - f2: wp.float32 (Refractive index in the -normal direction, ie. the inside)
   3: Aspheric lens. Requires:
    - v0: wp.vec3    (Origin of   axis)
    - v1: wp.vec3    (Normal of z-axis)
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
   4: Cylinder blocker. Requires:
    - v0: wp.vec3    (Center of basis)
    - v1: wp.vec3    (Normal of basis with height as length)
    - f0: wp.float32 (Radius)
"""
@wp.struct
class Primitive3D:
    type: wp.int32
    v0  : wp.vec3
    v1  : wp.vec3
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

@wp.struct
class Intersection3D:
    hit      : wp.bool
    hitPoint : wp.vec3
    normal   : wp.vec3 # surface normal at intersection (pointing outside)
    ray      : Ray3D
    primitive: Primitive3D


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

"""
   Types:
   ------
   0: Pointlight. Requires:
    - v0: wp.vec3  (Origin)
    - f0: wp.float (Intensity)
   1: Parallelogram Lambertian. Requires:
    - v0: wp.vec3 (Center) 
    - v1: wp.vec3 (Tangent)   # NOT normalized. Its length gives the length of the light source
    - v2: wp.vec3 (Bitangent) # NOT normalized. Its length gives the width  of the light source
      # The normal will be then computed from the two first tangent and bitangent vectors.
    - f0: wp.float32 (Intensity)
"""
@wp.struct
class LightSource3D:
    type: wp.int32
    v0  : wp.vec3
    v1  : wp.vec3
    v2  : wp.vec3
    f0  : wp.float32

"""
   Types:
   0: Ideal sensor. Requires:
    - v0: wp.vec2  (Origin)
    - v1: wp.vec2  (End)
    - i0: wp.int32 (Number of pixels)
   1: Ideal Shack-Hartmann sensor. Requires:
    - v0: wp.vec2  (Origin)
    - v1: wp.vec2  (End)
    - i0: wp.int32 (Number of pixels)
   2: Ideal plenoptic sensor. Requires:
    - v0: wp.vec2  (Origin)
    - v1: wp.vec2  (End)
    - i0: wp.int32 (Number of spatial bins, or "pixels")
    - i1: wp.int32 (Number of angular bins)
"""
@wp.struct
class Sensor:
    type: wp.int32
    v0  : wp.vec2
    v1  : wp.vec2
    i0  : wp.int32
    i1  : wp.int32

"""
   Types:
   0: Ideal 3D sensor. Requires:
    - v0: wp.vec3 (Center)
    - v1: wp.vec3 (Tangent)   # NOT normalized. Its length gives the length of the light source
    - v2: wp.vec3 (Bitangent) # NOT normalized. Its length gives the width  of the light source
      # The normal will be then computed from the two first tangent and bitangent vectors.
    - i0: wp.int32 (Number of pixels on the tangential axis)
      # Pixels are squares, so number of pixels on the bitangential axis is deduced from latter)
   1: Ideal plenoptic 3D sensor. Requires:
    - v0: wp.vec3 (Center)
    - v1: wp.vec3 (Tangent)   # NOT normalized. Its length gives the length of the light source
    - v2: wp.vec3 (Bitangent) # NOT normalized. Its length gives the width  of the light source
      # The normal will be then computed from the two first tangent and bitangent vectors.
    - i0: wp.int32 (Number of pixels on the tangential axis)
      # Pixels are squares, so number of pixels on the bitangential axis is deduced from latter)
    - i1: wp.int32 (Number of angular bins over zenithal  angle for each spatial pixel)
    - i2: wp.int32 (Number of angular bins over azimuthal angle for each spatial pixel)
"""
@wp.struct
class Sensor3D:
    type: wp.int32
    v0  : wp.vec3
    v1  : wp.vec3
    v2  : wp.vec3
    i0  : wp.int32
    i1  : wp.int32
    i2  : wp.int32