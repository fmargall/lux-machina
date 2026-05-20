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

@wp.struct
class _Material:
    iorPositive: wp.float32
    iorNegative: wp.float32

@wp.struct
class _Primitive:
    type      : wp.int32
    materialID: wp.int32
    v0        : wp.vec3f
    v1        : wp.vec3f
    v2        : wp.vec3f
    v3        : wp.vec3f

@wp.struct
class _Intersection:
    t          : wp.float32
    normal     : wp.vec3f
    primitiveID: wp.int32

@wp.struct
class _LightSource:
    type       : wp.int32
    primitiveID: wp.int32
    power      : wp.float32

@wp.struct
class _Sensor:
    type: wp.int32
    v0  : wp.vec3f
    v1  : wp.vec3f
    v2  : wp.vec3f
    v3  : wp.vec3f
    i0  : wp.int32
    i1  : wp.int32