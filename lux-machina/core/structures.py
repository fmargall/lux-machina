import warp as wp

"""
   Types
   -----

    - 1: Parallelogram Lambertian. Requires:
      - v0: wp.vec3f   (Origin)
      - v1: wp.vec3f   (Tangent)   # NOT normalized. Its length gives the length of the emitter
      - v2: wp.vec3f   (Bitangent) # NOT normalized. Its length gives the width  of the emitter
      - f0: wp.float32 (Intensity) 
      # Notes: The normal will be then computed from the two first tangent and bitangent vec3f.
"""
@wp.struct
class _Emitter3D:
    type: wp.int32
    v0  : wp.vec3f
    v1  : wp.vec3f
    v2  : wp.vec3f
    f0  : wp.float32


_OUT_IN = wp.int32(0)
_IN_OUT = wp.int32(1)

@wp.struct
class _Ray3D:
    isAlive   : wp.bool
    origin    : wp.vec3f
    direction : wp.vec3f
    depth     : wp.int32
    energy    : wp.float32
    wavelength: wp.float32
    ioState   = wp.int32

"""
   Types:
   ------

   - 0: Annulus blocker. Requires:
     - v0: wp.vec3f   (Origin)
     - v1: wp.vec3f   (Normal)
     - f0: wp.float32 (Inner radius)
     - f1: wp.float32 (Outer radius)

   - 1: Cylinder. Requires:
     - v0: wp.vec3f   (Center of basis)
     - v1: wp.vec3f   (Normal of basis with height as length)
     - f0: wp.float32 (Radius)
     - f1: wp.float32 (Refractive index in the +normal direction, ie. the outside)
     - f2: wp.float32 (Refractive index in the -normal direction, ie. the inside)

   - 2: Spherical cap. Requires:
     - v0: wp.vec3f   (Center of sphere)
     - v1: wp.vec3f   (Direction of spherical cap pole, with radius as length)
     - f0: wp.float32 (Angle of the spherical cap: pi(/2) for an (hemi)sphere)
     - f1: wp.float32 (Refractive index in the +normal direction, ie. the outside)
     - f2: wp.float32 (Refractive index in the -normal direction, ie. the inside)

   - 3: Aspheric lens. Requires:
     - v0: wp.vec3f   (Origin of   axis)
     - v1: wp.vec3f   (Normal of z-axis)
     - f0 : wp.float32 (Radius)
     - f1 : wp.float32 (Conic constant)
     - f2 : wp.float32 (Refractive index in the +normal direction)
     - f3 : wp.float32 (Refractive index in the -normal direction)
     - f4 : wp.float32 (y-intercept) 
     - f5 : wp.float32 (normalization factor)
     - f6 : wp.float32 (1st coefficient, associated to  2nd power)
     - f7 : wp.float32 (2nd coefficient, associated to  4th power)
     - f8 : wp.float32 (3rd coefficient, associated to  6th power)
     - f9 : wp.float32 (4th coefficient, associated to  8th power)
     - f10: wp.float32 (5th coefficient, associated to 10th power)

   - 4: Cylinder blocker. Requires:
     - v0: wp.vec3f   (Center of basis)
     - v1: wp.vec3f   (Normal of basis with height as length)
     - f0: wp.float32 (Radius)

   - 5: Parallelogram. Requires:
     - v0: wp.vec3f (Vertex 0, connected to 1 and 3)
     - v1: wp.vec3f (Vertex 1, connected to 0 and 2)
     - v2: wp.vec3f (Vertex 2, connected to 1 and 3)
     - v3: wp.vec3f (Vertex 3, connected to 0 and 2)

   - 6: Disk. Requires:
     - v0: wp.vec3f (Origin)
     - v1: wp.vec3f (Normal, with radius as length)
"""
@wp.struct
class _Primitive3D:
    type: wp.int32
    v0  : wp.vec3f
    v1  : wp.vec3f
    v2  : wp.vec3f
    v3  : wp.vec3f
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
class _Intersection3D:
    hit      : wp.bool
    hitPoint : wp.vec3f
    normal   : wp.vec3f # surface normal at intersection (pointing outside)
    ray      : _Ray3D
    primitive: _Primitive3D