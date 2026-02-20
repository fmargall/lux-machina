import warp as wp

@wp.func
def distanceOnUnitDisk(p1: wp.vec2, p2: wp.vec2) -> wp.float32:
    r1, t1 = p1[0], p1[1]
    r2, t2 = p2[0], p2[1]

    d2 = r1 * r1 + r2 * r2 - wp.float32(2.) * r1 * r2 * wp.cos(t1 - t2)
    return wp.sqrt(wp.max(d2, 0.0))

@wp.func
def distanceOnUnitHemisphere(p1: wp.vec2, p2: wp.vec2) -> wp.float32:
    theta1, phi1 = p1[0], p1[1]
    theta2, phi2 = p2[0], p2[1]

    cos_d = wp.cos(theta1) * wp.cos(theta2) + wp.sin(theta1) * wp.sin(theta2) * wp.cos(phi1 - phi2)
    return wp.acos(cos_d)