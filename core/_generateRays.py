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
def _sampleAreaLight(
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
        sourcePrimitiveID = wp.int32(-1)
    )

    return ray

@wp.func
def _samplePointLight(
    lightSource: _LightSource,
    N          : wp.int32,
    rngState   : wp.uint32
) -> _Ray:

    # Point light: origin = v0, cone axis = v1, half-aperture = f0.
    # f0 = pi corresponds to isotropic emission over the full sphere.
    origin    = lightSource.v0
    axis      = lightSource.v1
    halfAngle = lightSource.f0

    # ── Uniform solid-angle sampling inside the cone of half-angle f0 ──
    # cos(theta) uniform in [cos(f0), 1], phi uniform in [0, 2*pi)
    xi = wp.sample_unit_square(rngState)
    xi = xi + wp.vec2f(0.5, 0.5) # [0; 1]

    cosMin   = wp.cos(halfAngle)
    cosTheta = cosMin + xi[0] * (wp.float32(1.) - cosMin)
    # Clamp for numerical safety before sqrt (FP error near grazing)
    sinTheta = wp.sqrt(wp.max(wp.float32(0.),
                              wp.float32(1.) - cosTheta * cosTheta))
    phi      = wp.float32(2.) * wp.pi * xi[1]

    localDir = wp.vec3f(sinTheta * wp.cos(phi),
                        sinTheta * wp.sin(phi),
                        cosTheta)

    # Rotate from local frame (z = up) to world frame (z = axis)
    direction = _alignWithNormal(localDir, axis)

    # Cone solid angle:
    #   Omega = 2*pi * (1 - cos(f0))
    #   f0 = pi   -> Omega = 4*pi   (isotropic, full sphere)
    #   f0 = pi/2 -> Omega = 2*pi   (hemisphere)
    solidAngle   = wp.float32(2.) * wp.pi * (wp.float32(1.) - cosMin)
    pdfDirection = wp.float32(1.) / solidAngle

    # Intensity is uniform : I = phi / Omega (W/sr).
    # With direction pdf = 1/Omega, the MC estimator collapses to phi / N
    throughput = lightSource.power / wp.float32(N)

    ray = _Ray(
        isAlive           = True,
        origin            = origin,
        direction         = direction,
        throughput        = throughput,
        # Position is a Dirac delta and is folded into the
        # throughput so only the directional pdf is stored
        pdf               = pdfDirection,
        depth             = wp.int32(0),
        wavelength        = wp.float32(0.),
        sourcePrimitiveID = wp.int32(-1)
    )

    return ray

@wp.func
def _sampleGaussianBeam(
    lightSource: _LightSource,
    N          : wp.int32,
    rngState   : wp.uint32
) -> _Ray:

    origin = lightSource.v0
    axis   = lightSource.v1
    w0     = lightSource.f0

    # ── Position sampling in the waist plane ──
    sigmaPos = wp.float32(0.5) * w0
    localX   = sigmaPos * wp.randn(rngState)
    localY   = sigmaPos * wp.randn(rngState)

    # Move offset from local frame (z = up) to world frame (z = axis)
    localPos  = wp.vec3f(localX, localY, wp.float32(0.))
    rayOrigin = origin + _alignWithNormal(localPos, axis)

    # For now, the wavevelength is hardcoded to H-alpha, but it could
    # be sampled from a spectrum associated to the source more later.
    _WAVELENGTH = wp.float32(656.28e-9)

    # ── Direction sampling (paraxial Gaussian in angle) ──
    sigmaDir = _WAVELENGTH / (wp.float32(2.) * wp.pi * w0)
    thetaX   = sigmaDir * wp.randn(rngState)
    thetaY   = sigmaDir * wp.randn(rngState)

    # Paraxial approximation : local direction ≈ (theta_x, theta_y, 1)
    localDir  = wp.normalize(wp.vec3f(thetaX, thetaY, wp.float32(1.)))
    direction = _alignWithNormal(localDir, axis)

    # ── PDFs (position : per unit area ; direction : per unit solid angle) ──
    invTwoSigPos2 = wp.float32(1.) / (wp.float32(2.) * sigmaPos * sigmaPos)
    invTwoSigDir2 = wp.float32(1.) / (wp.float32(2.) * sigmaDir * sigmaDir)

    r2pos = localX * localX + localY * localY
    r2dir = thetaX * thetaX + thetaY * thetaY

    pdfPosition  = invTwoSigPos2 / wp.pi * wp.exp(-r2pos * invTwoSigPos2)
    pdfDirection = invTwoSigDir2 / wp.pi * wp.exp(-r2dir * invTwoSigDir2)

    # ── Throughput ──
    # Importance sampling matches the source emission distribution exactly
    # (position and direction sampled from their actual Gaussian profiles),
    # so the MC estimator collapses to phi / N -- same trick as the
    # lambertian and point-light samplers.
    throughput = lightSource.power / wp.float32(N)

    ray = _Ray(
        isAlive           = True,
        origin            = rayOrigin,
        direction         = direction,
        throughput        = throughput,
        # Position is a Dirac delta and is folded into the
        # throughput so only the directional pdf is stored
        pdf               = pdfPosition * pdfDirection,
        depth             = wp.int32(0),
        wavelength        = _WAVELENGTH,
        sourcePrimitiveID = wp.int32(-1)
    )

    return ray

@wp.func
def _sampleLightSource(
    lightSource: _LightSource,
    N          : wp.int32,
    rngState   : wp.uint32
) -> _Ray:
    # Dispatch on light source type
    # (see _LightSource docstring).

    if   lightSource.type == wp.int32(1): # Point light
        return _samplePointLight(lightSource, N, rngState)

    elif lightSource.type == wp.int32(2): # Gaussian beam
        return _sampleGaussianBeam(lightSource, N, rngState)

    # By default: type 0 => lambertian parallelogram
    return _sampleAreaLight(lightSource, N, rngState)

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