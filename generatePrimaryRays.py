import warp as wp

from structures import LightSource, LightSource3D, Ray, Ray3D

@wp.func
def generateRayFromPointLightSource(
    lightSource: LightSource,
    seed       : wp.int32,
) -> Ray:
    rand = wp.randf(wp.uint32(seed))
    theta = rand * 6.283185307179586
    
    ray = Ray()
    ray.origin    = lightSource.v0
    ray.direction = wp.vec2(wp.cos(theta), wp.sin(theta))
    ray.depth     = 0
    ray.energy    = lightSource.f0
    ray.isAlive   = True

    return ray

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
    
    ray = Ray3D()
    ray.origin    = lightSource.v0
    ray.direction = wp.vec3(wp.sin(theta) * wp.cos(phi), 
                            wp.sin(theta) * wp.sin(phi),
                            wp.cos(theta))
    ray.depth     = 0
    ray.energy    = lightSource.f0
    ray.isAlive   = True

    return ray

@wp.func
def generateRayFromLambertianSource(
    lightSource: LightSource,
    seed       : wp.int32,
) -> Ray:
    rand = wp.randf(wp.uint32(seed))

    ray = Ray()
    ray.origin = lightSource.v0 + (lightSource.v1 - lightSource.v0) * rand
    
    lightTangent = wp.normalize(lightSource.v1 - lightSource.v0)
    lightNormal  = wp.normalize(wp.vec2(-lightTangent.y, lightTangent.x))
    
    seed  = seed * 747796405 + 2891336453
    rand  = wp.randf(wp.uint32(seed))
    theta = wp.asin(2. * rand - 1.)
    
    ray.direction = wp.cos(theta) * lightNormal + wp.sin(theta) * lightTangent

    ray.depth   = 0
    ray.energy  = lightSource.f0
    ray.isAlive = True

    return ray

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

    ray = Ray3D()
    ray.origin = lightCenter + randX * lightTangent + randY * lightBitangent

    randTheta = wp.randf(wp.uint32(seedTheta))
    randPhi   = wp.randf(wp.uint32(seedPhi))

    # Hemisphere cosine-weighted sampling
    # since the source is Lambertian here
    theta = wp.acos(wp.sqrt(1. - randTheta))
    phi   = randPhi * 6.283185307179586
    
    direction = wp.sin(theta) * wp.cos(phi) * wp.normalize(lightTangent)   \
              + wp.sin(theta) * wp.sin(phi) * wp.normalize(lightBitangent) \
              + wp.cos(theta)               * wp.normalize(lightNormal)
    ray.direction = wp.normalize(direction)
    ray.depth     = 0
    ray.energy    = lightSource.f0
    ray.isAlive   = True

    return ray

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

    ray = Ray3D()
    ray.origin = lightCenter + randX * lightTangent + randY * lightBitangent
    ray.direction = wp.normalize(lightNormal)
    ray.depth     = 0
    ray.energy    = lightSource.f0
    ray.isAlive   = True

    return ray

@wp.kernel
def generatePrimaryRays(
    frameID            : wp.int32,
    lightSourcesBuffer : wp.array(dtype=LightSource, ndim=1),
    
    raysBuffer         : wp.array(dtype=Ray, ndim=1)
):
    # Get ray ID
    ID = wp.tid()

    # Pseudo-random seed generation
    seed = ID + frameID * raysBuffer.shape[0]

    # Selecting one light source
    # (For now always first one)
    lightSource = lightSourcesBuffer[0]

    if lightSource.type == 0: # Point light source
        ray = generateRayFromPointLightSource(lightSource, seed)
    if lightSource.type == 1: # Lambertian light source
        ray = generateRayFromLambertianSource(lightSource, seed)

    raysBuffer[ID] = ray

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

    if lightSource.type == 0: # Point light source
        ray = generateRayFrom3DPointLightSource(lightSource, seed)
    if lightSource.type == 1: # Lambertian light source
        ray = generateRayFrom3DLambertianSource(lightSource, seed)
    if lightSource.type == 2: # Collimated parallelogram light source
        ray = generateRayFrom3DCollimatedParallelogramSource(lightSource, seed)

    raysBuffer[ID] = ray