import numpy
from source import Source, diag
from nutils import debug

_ = numpy.newaxis


class MogiSource( Source ):

  def __init__( self, xyz, delta=0, azimuth=0 ):
    self.xyz = numpy.array(xyz)
    azimuth *= numpy.pi/180
    self.rotmat = numpy.array( [[numpy.cos(azimuth), numpy.sin(azimuth),0],[-numpy.sin(azimuth),numpy.cos(azimuth),0],[0,0,1]] )
    assert self.xyz.ndim == 1 and self.xyz.size == 3

  def displacement( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd = (xyz - self.xyz).dot( self.rotmat )
    R = numpy.linalg.norm( xyd, axis=-1 )
    uA_ = -xyd / R[...,_]**3

    xyd[...,2] -= xyz[...,2] * 2
    R = numpy.linalg.norm( xyd, axis=-1 )
    uA = -xyd / R[...,_]**3

    uC = 6 * xyd * xyd[...,2,_] / R[...,_]**5
    uC[...,2] *= -1
    uC[...,2] += 2. / R**3

    u = uA - uA_ + xyz[...,2,_] * uC
    u *= .25 / (1-poisson)
    u -= uA
    u *= 1 - 2*poisson
    return u.dot( self.rotmat.T )

  def gradient( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd = xyz - self.xyz
    R = numpy.linalg.norm( xyd, axis=-1 )

    duAdxyz_ = 3 * xyd[...,_,:] * (xyd/R[...,_]**5)[...,:,_]
    diag(duAdxyz_)[...] -= 1. / R[...,_]**3

    xyd[...,2] -= xyz[...,2] * 2
    R = numpy.linalg.norm( xyd, axis=-1 )

    duAdxyz = 3 * xyd[...,_,:] * (xyd/R[...,_]**5)[...,_]
    diag(duAdxyz)[...] -= 1. / R[...,_]**3
    duAdxyz[...,2] *= -1

    duCdxyz = -30 * xyd[...,_,:] * xyd[...,2,_,_] * (xyd/R[...,_]**7)[...,:,_]
    duCdxyz[...,2] += 6 * xyd / R[...,_]**5
    duCdxyz[...,2,:] *= -1
    duCdxyz[...,2,:] -= 6 * xyd / R[...,_]**5
    duCdxyz[...,2] *= -1
    diag(duCdxyz)[...] += 6 * (xyd[...,2]/R**5)[...,_]

    uC = 6 * xyd * (xyd[...,2]/R**5)[...,_]
    uC[...,2] *= -1
    uC[...,2] += 2. / R**3

    dudxyz = duAdxyz - duAdxyz_ + xyz[...,2,_,_] * duCdxyz
    dudxyz[...,2] += uC
    dudxyz *= .25 / (1-poisson)
    dudxyz -= duAdxyz
    dudxyz *= 1 - 2*poisson
    return dudxyz
