import numpy
from source import Source, diag

_ = numpy.newaxis
PI = numpy.pi
                                                                                                                                                                                                                                                            
class TensileSource( Source ):
  def __init__( self, xyz, delta=90, azimuth=0 ):
    self.xyz = numpy.asarray( xyz )
    assert self.xyz.ndim == 1 and self.xyz.size == 3
    
    self.delta = delta * PI / 180        
    self.azimuth   = azimuth * PI / 180
    self.cosd  = numpy.cos(self.delta)
    self.cos2d = numpy.cos(2.*self.delta)
    self.sind  = numpy.sin(self.delta)
    self.sin2d = numpy.sin(2.*self.delta)
    self.rotmat = numpy.array( [ [numpy.cos(self.azimuth),numpy.sin(self.azimuth),0] , [-numpy.sin(self.azimuth),numpy.cos(self.azimuth),0],[0,0,1] ] )
  
  def displacement( self, xyz, poisson=0.25 ):  
    xyz = numpy.asarray( xyz )
    
    xyd      = numpy.dot( xyz * [1,1,-1] - self.xyz, self.rotmat )
    xydprime = numpy.dot( xyz - self.xyz, self.rotmat )

    d = xyd[...,2]
    c = -self.xyz[2]*numpy.ones_like(d)

    alpha = .5 / (1 - poisson)

    R      = numpy.linalg.norm( xyd , axis=-1 )
    Rprime = numpy.linalg.norm( xydprime , axis=-1 )

    q      = xyd[...,1]*self.sind - xyd[...,2]*self.cosd
    qprime = xydprime[...,1]*self.sind - xydprime[...,2]*self.cosd
    p      = xyd[...,1]*self.cosd + xyd[...,2]*self.sind
    pprime = xydprime[...,1]*self.cosd + xydprime[...,2]*self.sind
    s      = p*self.sind + q*self.cosd
    sprime = pprime*self.sind + qprime*self.cosd
    t      = p*self.cosd - q*self.sind
    tprime = pprime*self.cosd - qprime*self.sind
      
    a1      = (0.5*(1-alpha) / R**3)[...,_]
    a1prime = (0.5*(1-alpha) / Rprime**3)[...,_]
    a2      = (1.5*(alpha*q**2) / R**5)[...,_]
    a2prime = (1.5*(alpha*qprime**2) / Rprime**5  )[...,_]
    a3      = ((3.0*q**2) / R**5)[...,_]
    a4      = ((1-alpha)*self.sind**2 / alpha  )[...,_]
    a5      = ((1-alpha) / R**3)[...,_]
    a6      = (alpha*3.*c / R**5)[...,_]
    a7      = (alpha*3.*xyz[...,2] / R**5)[...,_]
        
    I1 =  xyd[...,1] * ( ( 1. / (R * (R+xyd[...,2])**2) ) - xyd[...,0]**2*( (3.0*R+xyd[...,2]) / (R**3 * (R+xyd[...,2])**3) ) )
    I2 =  xyd[...,0] * ( ( 1. / (R*(R+xyd[...,2])**2) ) - xyd[...,1]**2*( (3.0*R+xyd[...,2])/(R**3 *(R+xyd[...,2])**3) ) )
    I3 =  xyd[...,0] / R**3 - I2
    I5 =  1. / ( R*(R+xyd[...,2]) ) - xyd[...,0]**2 * ( 2*R+xyd[...,2] ) / ( R**3 * (R+xyd[...,2])**2 )
  
    A3      = 1.0 - 3.0*xyd[...,0]**2 / R**2
          
    uA = a1 * numpy.array( [xyd[...,0], t, s] ).T \
       - a2 * xyd
    uAprime = a1prime * numpy.array( [xydprime[...,0], tprime, sprime] ).T \
            - a2prime * xydprime
    uB = a3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
       - a4 * numpy.array( [ I3 , I1 , I5 ] ).T
    uC = a5 * numpy.array( [-3*xyd[...,0]*s/R**2 , self.sin2d - 3*xyd[...,1]*s/R**2 , -(1 - A3*self.sind**2) ] ).T \
       + a6 * numpy.array( [5*xyd[...,0]*q**2/R**2 , t - xyd[...,1] + 5*xyd[...,1]*q**2/R**2 , -(s - d + 5*d*q**2/R**2)] ).T \
       + a7 * [-1,-1,1] * xyd 
            
    return numpy.dot( uA - uAprime + uB + xyz[...,2,_]*uC, self.rotmat.T )
    
    
  def gradient( self, xyz, poisson=0.25 ):
    xyz = numpy.asarray( xyz )
    
    xyd      = numpy.dot( xyz * [1,1,-1] - self.xyz, self.rotmat )
    xydprime = numpy.dot( xyz - self.xyz, self.rotmat )
    
    alpha  = .5 / ( 1 - poisson )
    R      = numpy.linalg.norm( xyd, axis=-1 )
    Rprime = numpy.linalg.norm( xydprime, axis=-1 )

    q      = xyd[...,1]*self.sind - xyd[...,2]*self.cosd
    qprime = xydprime[...,1]*self.sind - xydprime[...,2]*self.cosd
    p      = xyd[...,1]*self.cosd + xyd[...,2]*self.sind
    pprime = xydprime[...,1]*self.cosd + xydprime[...,2]*self.sind

    s      = p*self.sind + q*self.cosd
    sprime = pprime*self.sind + qprime*self.cosd
    t      = p*self.cosd - q*self.sind
    tprime = pprime*self.cosd - qprime*self.sind    

    A3      = 1.0 - 3.0*xyd[...,0]**2 / R**2   
    A3prime = 1.0 - 3.0*xydprime[...,0]**2 / Rprime**2   
    A5      = 1.0 - 5.0*xyd[...,0]**2 / R**2
    A5prime = 1.0 - 5.0*xydprime[...,0]**2 / Rprime**2
    A7      = 1.0 - 7.0*xyd[...,0]**2 / R**2
    
    B5 = 1-5. * xyd[...,1]**2 / R**2
    B7 = 1-7. * xyd[...,1]**2 / R**2
    
    C5 = 1-5. * xyd[...,2]**2 / R**2
    C7 = 1-7. * xyd[...,2]**2 / R**2
   
    d      = xyd[...,2]
    dprime = xydprime[...,2]
    c      = -self.xyz[2]*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0]*xyd[...,1] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0]**2*xyd[...,1]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    
    K1 =-xyd[...,1] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,0]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) ) 
    K2 =-xyd[...,0] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,1]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) )  
    K3 =-3.*xyd[...,0]*d/R**5 - K2
    
    U         = self.sind - 5*xyd[...,1]*q / R**2
    Uprime    = self.sind - 5*xydprime[...,1]*qprime / Rprime**2
    Upri      = self.cosd + 5*d*q / R**2
    Upriprime = self.cosd + 5*xydprime[...,2]*qprime / Rprime**2    
    
    W         = self.sind + U
    Wprime    = self.sind + Uprime
    Wpri      = self.cosd + Upri
    Wpriprime = self.cosd + Upriprime
    
    a1      = (0.5*(1-alpha) / R**3)[...,_]
    a1prime = (0.5*(1-alpha) / Rprime**3)[...,_]
    a2      = (1.5*alpha*q**2 / R**5)[...,_]
    a2prime = (1.5*alpha*qprime**2 / Rprime**5    )[...,_]
    a3      = (3.*q**2 / R**5        )[...,_]
    a4      = ((1-alpha)*self.sind**2/alpha)[...,_]
    a5      = ((1-alpha)*3. / R**5)[...,_]
    a6      = (alpha*15.*c/R**7)[...,_]
    a7      = (alpha*3.*xyz[...,2] / R**5)[...,_]
    
    b2      = (1.5*alpha*q*W/R**5)[...,_]
    b2prime = (1.5*alpha*qprime*Wprime / Rprime**5)[...,_]
    b3      = (3.*q*W / R**5)[...,_]
    b5      = (alpha*3.*c/R**5)[...,_]
    
    c2      = (1.5*alpha*q*Wpri / R**5)[...,_]
    c2prime = (1.5*alpha*qprime*Wpriprime / Rprime**5)[...,_]
    c3      = (3.*q*Wpri / R**5)[...,_]
    c6      = (alpha*3./R**5)[...,_]
    
    duAdx = a1 * numpy.array( [ A3 , -3.*xyd[...,0]*t/R**2, -3.*xyd[...,0]*s/R**2 ] ).T \
          + a2 * numpy.array( [ -A5 , 5.*xyd[...,0]*xyd[...,1]/R**2 , 5.*xyd[...,0]*d/R**2 ] ).T
    duAdxprime = a1prime * numpy.array( [ A3prime , -3.*xydprime[...,0]*tprime/Rprime**2, -3.*xydprime[...,0]*sprime/Rprime**2 ] ).T \
               + a2prime * numpy.array( [ -A5prime , 5.*xydprime[...,0]*xydprime[...,1]/Rprime**2 , 5.*xydprime[...,0]*dprime / Rprime**2] ).T
    duBdx = a3 * numpy.array( [ A5 , -5*xyd[...,0]*xyd[...,1]/R**2 , -5*c*xyd[...,0]/R**2 ] ).T \
          - a4 * numpy.array( [ J3, J1, K3] ).T

    duCdx = a5 * numpy.array( [ -s*A5 , -xyd[...,0]*(self.sin2d - 5*xyd[...,1]*s/R**2), \
                                 xyd[...,0]*(1 - (2+A5)*self.sind**2) ] ).T \
          + a6 * numpy.array( [ q**2*A7 , -xyd[...,0] * (t - xyd[...,1] + 7*xyd[...,1]*q**2/R**2) , \
                                 xyd[...,0] * (s - d + 7*d*q**2/R**2) ] ).T \
          + a7 * numpy.array( [-A5 , 5*xyd[...,0]*xyd[...,1] / R**2 , -5*xyd[...,0]*d / R**2] ).T
    duAdy = a1 * numpy.array( [ -3*xyd[...,0]*xyd[...,1]/R**2 , self.cos2d - 3*xyd[...,1]*t/R**2 , \
                                 self.sin2d - 3*xyd[...,1]*s/R**2 ] ).T \
          - b2 * xyd \
          - a2 * [ 0, 1, 0 ]                             
    duAdyprime = a1prime * numpy.array( [ -3*xydprime[...,0]*xydprime[...,1]/Rprime**2 , self.cos2d - 3*xydprime[...,1]*tprime/Rprime**2 , \
                                 self.sin2d - 3.*xydprime[...,1]*sprime/Rprime**2 ] ).T \
               - b2prime * xydprime \
               - a2prime * [ 0, 1, 0 ]                                                 
    duBdy = b3 * numpy.array( [ xyd[...,0], xyd[...,1], c ] ).T \
          + a3 * [ 0, 1, 0 ] \
          - a4 * numpy.array( [ J1, J2, K1 ] ).T
                                                                                                             
    duCdy = a5 * numpy.array( [-xyd[...,0] * (self.sin2d - 5*xyd[...,1]*s/R**2) , \
                                -(2*xyd[...,1]*self.sin2d + s*B5) , \
                                 xyd[...,1] * (1 - A5*self.sind**2) ]).T \
          + b5 * numpy.array( [-5*xyd[...,0] * (t - xyd[...,1] + 7.*xyd[...,1]*q**2/R**2) / R**2 , 
                                -(2*self.sind**2 + 10*xyd[...,1]*(t-xyd[...,1])/R**2 - 5*q**2*B7/R**2) , 
                                 (3.+A5)*self.sin2d - 5*xyd[...,1]*d*(2.-7.*q**2/R**2)/R**2 ] ).T\
          + a7 * numpy.array( [ 5*xyd[...,0]*xyd[...,1]/R**2, -B5, -5*xyd[...,1]*d/R**2] ).T          
    
    duAdz = a1 * numpy.array( [ 3.*xyd[...,0]*d/R**2 , -(self.sin2d - 3.*d*t/R**2) , self.cos2d + 3.*d*s/R**2 ] ).T\
          - c2 * xyd \
          + a2 * [ 0, 0 , 1] 
    duAdzprime = a1prime * numpy.array( [ 3.*xydprime[...,0]*dprime/Rprime**2 , -(self.sin2d - 3.*dprime*tprime/Rprime**2) , \
                                           self.cos2d + 3.*dprime*sprime/Rprime**2 ] ).T\
               - c2prime * xydprime \
               + a2prime * [ 0, 0 , 1] 
    duBdz = c3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c] ).T \
          + a4 * numpy.array( [ K3, K1, -A3/R**3] ).T   
    duCdz = a5 * numpy.array( [ -xyd[...,0] * (self.cos2d + 5*d*s/R**2) , d*B5*self.sin2d - xyd[...,1]*C5*self.cos2d, 
                                 -d * (1 - A5*self.sind**2) ] ).T \
          + b5 * numpy.array( [ 5*xyd[...,0] * (s - d + 7*d*q**2/R**2)/R**2 , \
                                 (3+A5)*self.sin2d - 5*xyd[...,1]*d*(2-7*q**2/R**2)/R**2 , \
                                -(self.cos2d + 10*d*(s-d)/R**2 - 5*q**2*C7/R**2) ]).T \
          - c6 * numpy.array( [ xyd[...,0]*(1 + 5*d*xyz[...,2]/R**2) , xyd[...,1]*(1+5*d*xyz[...,2]/R**2), \
                                 xyz[...,2]*(1+C5) ]).T

    uC = a5 * numpy.array( [-xyd[...,0]*s , (R**2/3)*(self.sin2d - 3*xyd[...,1]*s/R**2) , 
                             -(R**2/3)*(1 - A3*self.sind**2) ] ).T \
       + b5 * numpy.array( [5*xyd[...,0]*q**2 /R**2 , (t - xyd[...,1] + 5*xyd[...,1]*q**2/R**2) , -(s - d + 5*d*q**2/R**2)] ).T \
       + a7 * [-1,-1,1] * xyd
   
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz+ uC
    
    G = numpy.array( [ dudx.T, dudy.T, dudz.T ]).T.swapaxes(-2,-1)
    return numpy.dot( self.rotmat, numpy.dot( G, self.rotmat.T ) ).swapaxes(0,1)
    
