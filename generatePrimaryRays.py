import warp as wp

from structures import LightSource, Ray

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