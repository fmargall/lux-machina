import warp as wp


@wp.struct
class _Ray:
    origin    : wp.vec3f
    direction : wp.vec3f
    throughput: wp.float32
    pdf       : wp.float32
    wavelength: wp.float32

    isAlive   : wp.bool
    depth     : wp.int32

    # Avoid self intersections
    sourcePrimitiveID: wp.int32

@wp.struct
class _Material:
    iorPositive: wp.float32
    iorNegative: wp.float32

"""
   Types:
   0 : Triangle
   1 : Quad
   2 : Disk (or annulus)
   3 : Sphere (or spherical cap)
       ─────────────────────────
       v0 (wp.vec3f)  : Center
       v1 (wp.vec3f)  : Direction of spherical cap pole (normalized)
       f0 (wp.float32): Radius
       f1 (wp.float32): Angle of the spherical cap: pi(/2) for an (hemi)sphere
   4 : Cylinder
       ────────
       v0 (wp.vec3f): Center of basis
       v1 (wp.vec3f): Axis (normalized)
       f0 (wp.vec3f): Radius basis
       f1 (wp.vec3f): Height

   5 : Asphere
"""
@wp.struct
class _Primitive:
    type      : wp.int32
    materialID: wp.int32
    v0        : wp.vec3f
    v1        : wp.vec3f
    v2        : wp.vec3f
    v3        : wp.vec3f
    f0        : wp.float32
    f1        : wp.float32
    f2        : wp.float32
    f3        : wp.float32
    f4        : wp.float32
    f5        : wp.float32
    f6        : wp.float32

@wp.func
def _isFlat(primitive: _Primitive) -> wp.bool:
    #            Triangle == 0              Quad == 1
    #        Disk/Annulus == 2
    return primitive.type == 0 or primitive.type == 1 \
        or primitive.type == 2

@wp.struct
class _Intersection:
    t          : wp.float32
    normal     : wp.vec3f
    primitiveID: wp.int32

@wp.struct
class _LightSource:
    type : wp.int32
    v0   : wp.vec3f
    v1   : wp.vec3f
    v2   : wp.vec3f
    v3   : wp.vec3f
    power: wp.float32

@wp.struct
class _Sensor:
    type: wp.int32
    v0  : wp.vec3f
    v1  : wp.vec3f
    v2  : wp.vec3f
    v3  : wp.vec3f
    i0  : wp.int32
    i1  : wp.int32