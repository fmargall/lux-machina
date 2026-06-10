from calendar import c

import warp as wp

from ..core._structures import _LightSource, _Primitive
from ..core._utils      import _rodriguesMatrix


class EdmundOptics:
    @staticmethod
    def EO88598(position: tuple = (0, 0, 0),
                rotation: tuple = (0, 0, 0)):
        pass

class Thorlabs:
    @staticmethod
    def ACL25416U(position: tuple = (0, 0, 0),
                  rotation: tuple = (0, 0, 0)):

        # First surface consists of a spherical cap
        convex = _Primitive()
        convex.type = wp.int32(3) # Spherical cap
        convex.v0   = wp.float32(0., 0., 0.06999948) # Center (in meters)
        convex.v1   = wp.float32(0., 0., -1.)        # Direction of spherical cap pole (normalized)
        convex.f0   = wp.float32(0.06999948)         # Radius (in meters)
        convex.f1   = wp.float32(0.182440)           # Angle of the spherical cap (in radians)

        # Then follows a cylinder that represents the border
        cylinder = _Primitive()
        cylinder.type = wp.int32(4) # Cylinder
        cylinder.v0   = wp.float32(0., 0., 0.00116158) # Origin of basis (in meters)
        cylinder.v1   = wp.float32(0., 0., 1.)         # Axis (normalized)
        cylinder.f0   = wp.float32(0.01270)            # Radius (in meters)
        cylinder.f1   = wp.float32(0.0012)             # Height (in meters)

        # Then follows an aspheric surface
        aspheric = _Primitive()
        aspheric.type = 5 # Asphere
        aspheric.v0 = wp.vec3f(0., 0., 0.01396158) # Local axis origin
        aspheric.v1 = wp.vec3f(0., 0., -0.001)     # (z-axis unit vector in the local frame)
        aspheric.f0 = wp.float32(8.818197)         # Radius
        aspheric.f1 = wp.float32(-0.9991715)       # Conic constant
        aspheric.f2 = wp.float32(8.682167e-5)      # 1st coefficient, associated to 4th power
        aspheric.f3 = wp.float32(6.3760123e-8)     # 2nd coefficient, associated to 6th power
        aspheric.f4 = wp.float32(2.4073084e-9)     # 3rd coefficient, associated to 8th power
        aspheric.f5 = wp.float32(-1.7189021e-11)   # 4th coefficient, associated to 10th power
        aspheric.f6 = wp.float32(12.7)             # Maximum radius (millimeters)

    @staticmethod
    def AC254030(position: tuple = (0, 0, 0),
                 rotation: tuple = (0, 0, 0)):
        pass

    @staticmethod
    def AC508075(position: tuple = (0, 0, 0),
                 rotation: tuple = (0, 0, 0)):
        pass

    @staticmethod
    def LB1723(position: tuple = (0, 0, 0),
               rotation: tuple = (0, 0, 0)):

        convex0 = _Primitive()
        convex0.type = wp.int32(3) # Spherical cap
        convex0.v0   = wp.float32(0., 0., 0.0592) # Center (in meters)
        convex0.v1   = wp.float32(0., 0., -1.)    # Direction of spherical cap pole (normalized)
        convex0.f0   = wp.float32(0.0592)         # Radius (in meters)
        convex0.f1   = wp.float32(0.447189)       # Angle of the spherical cap (in radians)

        cylinder = _Primitive()
        cylinder.type = wp.int32(4) # Cylinder
        cylinder.v0   = wp.float32(0., 0., 0.00147368216774) # Origin of basis (in meters)
        cylinder.v1   = wp.float32(0., 0., 1.)               # Axis (normalized)
        cylinder.f0   = wp.float32(0.0254)                   # Radius (in meters)
        cylinder.f1   = wp.float32(0.0030)                   # Height (in meters)

        convex1 = _Primitive()
        convex1.type = wp.int32(3) # Spherical cap
        convex1.v0   = wp.float32(0., 0., -0.0448) # Center (in meters)
        convex1.v1   = wp.float32(0., 0., 1.)      # Direction of spherical cap pole (normalized)
        convex1.f0   = wp.float32(0.0592)          # Radius (in meters)
        convex1.f1   = wp.float32(0.447189)        # Angle of the spherical cap (in radians)

    @staticmethod
    def LB1757(position: tuple = (0, 0, 0),
               rotation: tuple = (0, 0, 0)):
        pass

    @staticmethod
    def SOLIS1D(position: tuple = (0, 0, 0),
                rotation: tuple = (0, 0, 0)):

        # Adding the LED as a square Lambertian light source
        led = _LightSource()
        led.type  = wp.int32(0) # Lambertian parallelogram
        led.v0    = wp.float32(0., 0., 0.)  # Origin (in meters)
        led.v0    = wp.vec3f(-0.0015, -0.0015, 0.) # (in meters)
        led.v1    = wp.vec3f( 0.0015, -0.0015, 0.) # (in meters)
        led.v2    = wp.vec3f( 0.0015,  0.0015, 0.) # (in meters)
        led.v3    = wp.vec3f(-0.0015,  0.0015, 0.) # (in meters)
        led.power = wp.float32(8.75) # Total emmitted power (8.75 Watts at maximum current)

        # Adding the first element: an aspheric ACL25416U lens

        # Adding the second element: a biconvex LB1723 lens

        # Adding the two retaining rings before and after the LB1723 lens