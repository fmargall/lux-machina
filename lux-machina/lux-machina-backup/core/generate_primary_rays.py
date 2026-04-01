import warp as wp

from .structures import _Emitter3D, _IN_OUT, _Ray3D

@wp.func
def _generateRayFrom3DLambertianEmitter(
    emitter: _Emitter3D,
    seed   :  wp.int32
) -> _Ray3D:
    seedSquare = 478413 * seed + 34841534
    seedTheta  = 654612 * seed + 65841231
    seedPhi    = 851524 * seed + 23132452

    lightCenter    = emitter.v0
    lightTangent   = emitter.v1
    lightBitangent = emitter.v2
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

    return _Ray3D(
        isAlive    = True,
        origin     = origin,
        direction  = direction,
        depth      = wp.int32(0),
        energy     = emitter.f0,
        wavelength = wp.float32(0.),
        ioState    = _IN_OUT
    )

@wp.kernel
def _generatePrimaryRays(
    emittersBuffer: wp.array(dtype=_Emitter3D, ndim=1), 
    frameID       : wp.int32,

    raysBuffer    : wp.array(dtype=_Ray3D, ndim=1)
):
    # Gets ray ID
    ID = wp.tid()

    # Pseudo-random seed from frame/ray IDs
    seed = ID + frameID * raysBuffer.shape[0]

    # Sampling among list of emitters. (For now
    # only the first one is taken : this should
    # be enhanced, related to issue #1)
    emitter = emittersBuffer[0]

    if emitter.type == 1:
        ray = _generateRayFrom3DLambertianEmitter(emitter, seed)

    raysBuffer[ID] = ray