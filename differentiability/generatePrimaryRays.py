import warp as wp

from structures import LightSource3D, Ray3D

@wp.func
def generateRayFrom3DPointLightSource(
    lightSource: LightSource3D,
    seed       : wp.int32,
) -> Ray3D:
    
    seedTheta = 532654 * seed + 45615685
    seedPhi   = 203115 * seed + 15314848
    randTheta = wp.randf(wp.uint32(seedTheta))
    randPhi   = wp.randf(wp.uint32(seedPhi))
    
    theta = randTheta * 1.570796326794896
    phi   = randPhi   * 6.283185307179586
    
    origin    = lightSource.v0
    direction = wp.vec3(wp.sin(theta) * wp.cos(phi), 
                        wp.sin(theta) * wp.sin(phi),
                        wp.cos(theta))

    return Ray3D(
        origin    = origin,
        direction = direction,
        depth     = 0,
        energy    = lightSource.f0,
        isAlive   = True
    )

@wp.func
def generateRayFrom3DLambertianSource(
    lightSource: LightSource3D,
    seed       : wp.int32,
) -> Ray3D:
    
    seedSquare = 478413 * seed + 34841534
    seedTheta  = 654612 * seed + 65841231
    seedPhi    = 851524 * seed + 23132452

    lightCenter    = lightSource.v0
    lightTangent   = lightSource.v1
    lightBitangent = lightSource.v2
    lightNormal    = wp.cross(wp.normalize(lightTangent), wp.normalize(lightBitangent))

    randSquare = wp.sample_unit_square(wp.uint32(seedSquare))
    randX = randSquare.x # (in [-0.5 ; +0.5])
    randY = randSquare.y # (in [-0.5 ; +0.5])

    origin = lightCenter + randX * lightTangent + randY * lightBitangent

    randTheta = wp.randf(wp.uint32(seedTheta))
    randPhi   = wp.randf(wp.uint32(seedPhi))

    # Hemisphere cosine-weighted sampling
    # since the source is Lambertian here
    theta = wp.acos(wp.sqrt(1. - randTheta))
    phi   = randPhi * 6.283185307179586
    
    direction = wp.sin(theta) * wp.cos(phi) * wp.normalize(lightTangent)   \
              + wp.sin(theta) * wp.sin(phi) * wp.normalize(lightBitangent) \
              + wp.cos(theta)               * wp.normalize(lightNormal)
    direction = wp.normalize(direction)

    return Ray3D(
        origin    = origin,
        direction = direction,
        depth     = 0,
        energy    = lightSource.f0,
        isAlive   = True
    )

@wp.func
def generateRayFrom3DCollimatedParallelogramSource(
    lightSource: LightSource3D,
    seed       : wp.int32,
) -> Ray3D:
    
    seedSquare = 415365 * seed + 16532131

    lightCenter    = lightSource.v0
    lightTangent   = lightSource.v1
    lightBitangent = lightSource.v2
    lightNormal    = wp.cross(wp.normalize(lightTangent), wp.normalize(lightBitangent))

    randSquare = wp.sample_unit_square(wp.uint32(seedSquare))
    randX = randSquare.x # (in [-0.5 ; +0.5])
    randY = randSquare.y # (in [-0.5 ; +0.5])

    origin = lightCenter + randX * lightTangent + randY * lightBitangent

    return Ray3D(
        origin    = origin,
        direction = wp.normalize(lightNormal),
        depth     = 0,
        energy    = lightSource.f0,
        isAlive   = True
    )


@wp.kernel
def generatePrimary3DRays(
    frameID            : wp.int32,
    lightSourcesBuffer : wp.array(dtype=LightSource3D, ndim=1),

    raysBuffer         : wp.array(dtype=Ray3D, ndim=1)
):
    # Get ray ID
    ID = wp.tid()

    # Pseudo-random seed generation
    seed = ID + frameID * raysBuffer.shape[0]

    # Selecting one light source
    # (For now always first one)
    lightSource = lightSourcesBuffer[0]

    if   lightSource.type == 0: # Point light source
        ray = generateRayFrom3DPointLightSource(lightSource, seed)
    elif lightSource.type == 1: # Lambertian light source
        ray = generateRayFrom3DLambertianSource(lightSource, seed)
    elif lightSource.type == 2: # Collimated parallelogram light source
        ray = generateRayFrom3DCollimatedParallelogramSource(lightSource, seed)

    raysBuffer[ID] = ray