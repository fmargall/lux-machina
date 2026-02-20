import warp as wp

goldenRatio = wp.float32(1.6180339)

@wp.func
def fibonacciDiskSampling(n: wp.int32, N: wp.int32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    r   = wp.sqrt(n / N)
    psi = 2. * wp.pi * n / goldenRatio

    return wp.vec2(r, psi)

@wp.func
def fibonacciHemisphereSampling(n: wp.int32, N: wp.int32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    theta = wp.acos(1. - 2. * n / N) / 2.
    phi   = 2. * wp.pi * n / goldenRatio
    
    return wp.vec2(theta, phi)

@wp.func
def fibonacciHemisphericalCapSampling(n: wp.int32, N: wp.int32, thetaMax: wp.float32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    theta = wp.acos(1. - n * (1. - wp.cos(2. * thetaMax))) / 2.
    phi   = 2. * wp.pi * n / goldenRatio

    return wp.vec2(theta, phi)