import warp as wp

from intersections import intersect3DRayWithParallelogram

from structures import CameraModel, Intersection3D, Primitive3D, Ray3D

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


@wp.kernel
def accumulateCameraAfterLambertianPlate(
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    camera             : CameraModel,
    lambertianPlate    : Primitive3D,

    cameraBuffer       : wp.array(dtype=wp.float32, ndim=2)
):
    rayID = wp.tid()

    intersection = intersectionsBuffer[rayID]
    ray          = intersection.ray

    accumulate = False

    intersectPlate = intersect3DRayWithParallelogram(ray, lambertianPlate)
    if intersectPlate.hit:

        # The infinite ray intersects the plate parallelogram.
        # We need to check if another intersection has occured
        # before reaching the lambertian plate.
        if intersectionsBuffer[rayID].hit:
            # Is the intersection point before the lambertian plate along the ray?
            distToHit   = wp.norm_l2(intersectionsBuffer[rayID].hitPoint - ray.origin)
            distToPlate = wp.norm_l2(intersectPlate.hitPoint - ray.origin)

            if distToPlate < distToHit:
                accumulate = True

        else:
            accumulate = True

        # We will suppose that the camera sees the plate directly
        # So there is no need to check once again for another new
        # intersection
        if accumulate:
            plateTangent   = wp.normalize(lambertianPlate.v1 - lambertianPlate.v0)
            plateBitangent = wp.normalize(lambertianPlate.v3 - lambertianPlate.v0)
            plateNormal    = wp.normalize(wp.cross(plateTangent, plateBitangent))

            newRayDirection = wp.normalize(camera.v0 - intersectPlate.hitPoint)

            cosI = wp.dot(-ray.direction, plateNormal)
            if cosI < wp.float32(0.):
                cosI = wp.dot(-ray.direction, -plateNormal)

            proj = projectPointWorldToOpenCVPinholeCamera(
                intersectPlate.hitPoint, camera)

            if proj[2] > 0.5: # Visibility test

                # Bilinear splatting to allow differentiability
                x = proj[0]
                y = proj[1]

                i0 = wp.int32(wp.floor(x))
                j0 = wp.int32(wp.floor(y))
                i1 = i0 + 1
                j1 = j0 + 1

                dx = x - wp.float32(i0)
                dy = y - wp.float32(j0)

                w00 = (1.0 - dx) * (1.0 - dy)
                w10 = dx * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w11 = dx * dy

                e = ray.energy * cosI / wp.pi # Because of the Lambertian reflectance

                width  = camera.i0
                height = camera.i1

                if i0 >= 0 and i0 < width and j0 >= 0 and j0 < height:
                    wp.atomic_add(cameraBuffer, j0, i0, w00 * e)

                if i1 >= 0 and i1 < width and j0 >= 0 and j0 < height:
                    wp.atomic_add(cameraBuffer, j0, i1, w10 * e)

                if i0 >= 0 and i0 < width and j1 >= 0 and j1 < height:
                    wp.atomic_add(cameraBuffer, j1, i0, w01 * e)

                if i1 >= 0 and i1 < width and j1 >= 0 and j1 < height:
                    wp.atomic_add(cameraBuffer, j1, i1, w11 * e)
