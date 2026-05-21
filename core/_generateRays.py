import warp as wp

from _structures import _LightSource, _Primitive, _Ray

@wp.func
def _alignWithNormal(localDirection: wp.vec3f, normal: wp.vec3f) -> wp.vec3f:
    # Surely this can be optimized in the future
    # following (Frisvad 2012, Pixar's ONB 2017)
    if wp.abs(normal[0]) > wp.float32(0.9):
        helper = wp.vec3f(0., 1., 0.)
    else:
        helper = wp.vec3f(1., 0., 0.)

    tangent   = wp.normalize(wp.cross(helper, normal))
    bitangent = wp.cross(normal, tangent)

    return localDirection[0] * tangent   \
         + localDirection[1] * bitangent \
         + localDirection[2] * normal

@wp.func
def _sampleLightSource(
    lightSource: _LightSource,
    N          : wp.int32,
    rngState   : wp.uint32
) -> _Ray:

    # For now, we only sample parallelogram
    uv = wp.sample_unit_square(rngState)
    uv = uv + wp.vec2f(0.5, 0.5) # [0; 1]

    edge1 = lightSource.v1 - lightSource.v0
    edge2 = lightSource.v3 - lightSource.v0
    point = lightSource.v0 + uv[0] * edge1 + uv[1] * edge2

    # Geometric normal of the parallelogram
    crossE = wp.cross(edge1, edge2)
    area   = wp.length(crossE)
    normal = crossE / area

    # Position PDF (uniform on surface)
    pdfPosition = wp.float32(1.) / area

    # Sample lambertian emission direction
    xi = wp.sample_unit_square(rngState)
    xi = xi + wp.vec2f(0.5, 0.5) # [0; 1]

    # Sample in local hemisphere where z = up
    r        = wp.sqrt(xi[0])
    theta    = wp.float32(2.) * wp.pi * xi[1]
    localDir = wp.vec3f(r * wp.cos(theta),
                        r * wp.sin(theta),
                        wp.sqrt(wp.float32(1.) - xi[0]))

    # Rotating the output direction, from local
    # frame (z = up) to world frame (z = normal)
    direction = _alignWithNormal(localDir, normal)

    # Direction PDF (cosine-weighted hemisphere)
    cosTheta     = wp.dot(direction, normal)
    pdfDirection = cosTheta / wp.pi

    # Compute initial throughput. For a lambertian source
    # of total power phi, sampled with uniform-position +
    # cosine-weighted-direction, everything simplifies to
    # throughput = phi / N
    throughput = lightSource.power / wp.float32(N)

    ray = _Ray(
        isAlive    = True,
        origin     = point,
        direction  = direction,
        throughput = throughput,
        pdf        = pdfPosition * pdfDirection,
        depth      = wp.int32(0),
        # No spectral rendering yet
        wavelength = wp.float32(0.),
        # No last primitive info yet
        sourcePrimitiveID = wp.int32(-1),
        sourceConvexSide  = True
    )

    return ray

@wp.kernel
def _generateRays(
    # --- Scene data ---
    lightSourceArray: wp.array(dtype=_LightSource),
    # Optional : This will be added later
    # sourceCDF: wp.array(dtype=wp.float32),

    # --- RNG control ---
    inputSeed: wp.int32,
    frameID  : wp.int32,

    # --- Output buffer ---
    rayArray: wp.array(dtype=_Ray)
):
    # Get ray ID
    ID = wp.tid()
    N  = rayArray.shape[0]

    # Pseudo-random state generation
    rngState = wp.rand_init(inputSeed, ID + frameID * N)

    # Selecting one light source
    # (For now always first one)
    lightSource = lightSourceArray[0]

    rayArray[ID] = _sampleLightSource(lightSource, N, rngState)