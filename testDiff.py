import warp  as wp

from structures import LightSource3D, Sensor3D


if __name__ == "__main__":
    wp.init()
    wp.config.verbose = True                    # Better to check potentiel problems
    wp.config.verify_autograd_array_access=True # Better to check potential problems

    # Initialisation
    tz = wp.float32(0.)

    # Since optParams is the array that will be optimised
    # at the end, it is thus needed to set 'requires_grad' 
    # as True
    optParams = wp.array(
        [tz], dtype=wp.float32, requires_grad=True
    )

    # Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 1 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.003, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.003, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesList = [lightSource]

    # Initializing buffers
    emitterLightSourcesLocalBuffer = wp.array(    lightSourcesList , dtype=LightSource3D)
    emitterLightSourcesWorldBuffer = wp.zeros(len(lightSourcesList), dtype=LightSource3D)

    # Sensor initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0, 0.0, 0.5)
    sensor.v1 = wp.vec3(0.0, 0.1, 0.0)
    sensor.v2 = wp.vec3(0.1, 0.0, 0.0)
    sensor.i0 = wp.int32(512)

    # Reference initialization
    referenceBuffer = wp.zeros((512, 512), dtype=wp.float32)