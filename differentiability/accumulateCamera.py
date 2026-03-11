import warp as wp

from structures import CameraModel

@wp.func
def projectPointWorldToOpenCVPinholeCamera(
    pointWorld: wp.vec3f, camera: CameraModel
) -> wp.vec3f: # -> (u, v, isVisible)

    width  = camera.i0
    height = camera.i1

    # Extrinsics
    rotation = camera.m0
    position = camera.v0

    # Intrinsics
    fx = camera.f0
    fy = camera.f1
    cx = camera.f2
    cy = camera.f3

    # Radial distortion
    k1 = camera.f4
    k2 = camera.f5
    k3 = camera.f6
    k4 = camera.f7
    k5 = camera.f8
    k6 = camera.f9

    # Tangential distortion
    p1 = camera.f10
    p2 = camera.f11

    # Thin prism distortion
    s1 = camera.f12
    s2 = camera.f13
    s3 = camera.f14
    s4 = camera.f15

    # World to camera coordinates
    p = rotation * (pointWorld - position)

    # Check if the point is behind camera
    if p[2] <= wp.float32(0.):
        return wp.vec3f(-1., -1., 0.)

    invZ = wp.float32(1.) / p[2]
    
    x = p[0] * invZ
    y = p[1] * invZ

    # Radial distortion
    r2 =  x *  x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radialNumerator   = wp.float32(1.) + k1 * r2 + k2 * r4 + k3 * r6
    radialDenominator = wp.float32(1.) + k4 * r2 + k5 * r4 + k6 * r6
    radial = radialNumerator / radialDenominator

    xD = x * radial
    yD = y * radial

    # Tangential distortion
    xy = xD * yD

    xT = wp.float32(2.) * p1 * xy  + p2 * (r2 + wp.float32(2.) * xD * xD)
    yT = p1 * (r2 + wp.float32(2.) * yD * yD) + wp.float32(2.) * p2 * xy

    xD += xT
    yD += yT

    # Thin prism distortion
    xD += s1 * r2 + s2 * r4
    yD += s3 * r2 + s4 * r4

    # Intrinsics
    u = fx * xD + cx
    v = fy * yD + cy

    # Check sensor bounds
    if u < wp.float32(0.) or u >= wp.float32(width):
        return wp.vec3(-1., -1., 0.)

    if v < wp.float32(0.) or v >= wp.float32(height):
        return wp.vec3(-1., -1., 0.)


    return wp.vec3f(u, v, 1.)