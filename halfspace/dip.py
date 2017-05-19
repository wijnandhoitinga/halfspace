import numpy
from source import Source

_ = numpy.newaxis
PI = numpy.pi

class DipSource( Source ):
  def __init__( self, xyz, delta=90, azimuth=0 ):
    self.xyz = numpy.asarray( xyz )
    assert self.xyz.ndim == 1 and self.xyz.size == 3
    
    self.delta = delta * PI / 180        
    self.azimuth   = azimuth * PI / 180
    self.cosd  = numpy.cos(self.delta)
    self.cos2d = numpy.cos(2.*self.delta)
    self.sind  = numpy.sin(self.delta)
    self.sin2d = numpy.sin(2.*self.delta)
    self.rotmat = numpy.array( [ [numpy.cos(self.azimuth),-numpy.sin(self.azimuth),0] , [numpy.sin(self.azimuth),numpy.cos(self.azimuth),0],[0,0,1] ] )
    
  def displacement( self, xyz, poisson=0.25 ):
    xyz = numpy.asarray( xyz )
      
    xyd      = numpy.dot( xyz * [1,1,-1] - self.xyz, self.rotmat)
    xydprime = numpy.dot( xyz - self.xyz, self.rotmat)

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
    
    
    a1      = (0.5 * ( 1 - alpha ) / R**3)[...,_]
    a1prime = (0.5 * ( 1 - alpha ) / Rprime**3)[...,_]
    a2      = (1.5 * ( alpha*q*p ) / R**5)[...,_]
    a2prime = (1.5 * ( alpha*qprime*pprime ) / Rprime**5)[...,_]
    a3      = (( 3.0*p*q ) / R**5)[...,_]
    a4      = (( 1-alpha ) * self.sind * self.cosd / alpha)[...,_]
    a5      = (( 1-alpha ) / R**3)[...,_]
    a6      = (alpha*3*c / R**5)[...,_]
    
    I1 =  xyd[...,1] * ( ( 1. / (R * (R+xyd[...,2])**2) ) - xyd[...,0]**2*( (3.0*R+xyd[...,2]) / (R**3 * (R+xyd[...,2])**3) ) )
    I2 =  xyd[...,0] * ( ( 1. / (R*(R+xyd[...,2])**2) ) - xyd[...,1]**2*( (3.0*R+xyd[...,2])/(R**3 *(R+xyd[...,2])**3) ) )
    I3 =  xyd[...,0] / R**3 - I2
    I5 =  1. / ( R*(R+xyd[...,2]) ) - xyd[...,0]**2 * ( 2*R+xyd[...,2] ) / ( R**3 * (R+xyd[...,2])**2 )

    A3      = 1.0 - 3.0*xyd[...,0]**2 / R**2
     
    uA =  a1 * numpy.array( [numpy.zeros_like(s), s, -t] ).T \
       +  a2 * xyd
    uAprime =  a1prime * numpy.array( [numpy.zeros_like(sprime), sprime, -tprime] ).T \
            +  a2prime * xydprime            
    uB = -a3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
       +  a4 * numpy.array( [I3, I1, I5] ).T  
        
    uC =  a5 * numpy.array( [3*xyd[...,0]*t / R**2, \
                            -(self.cos2d - 3*xyd[...,1]*t / R**2), \
                            -A3*self.sind *self.cosd] ).T \
       +  a6 * numpy.array( [-5*xyd[...,0]*p*q/R**2, s-5*xyd[...,1]*p*q/R**2, t+5*d*p*q/R**2] ).T
   
    return numpy.dot( uA - uAprime + uB + xyz[...,2,_]*uC, self.rotmat.T )
    
  def gradient( self, xyz, poisson=0.25 ):
    xyz = numpy.asarray( xyz )
    
    xyd      = numpy.dot( xyz * [1,1,-1] - self.xyz, self.rotmat )
    xydprime = numpy.dot( xyz - self.xyz, self.rotmat)
    

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

    A3      = (1.0 - 3.0*xyd[...,0]**2 / R**2   )
    A5      = (1.0 - 5.0*xyd[...,0]**2 / R**2)
    A5prime = (1.0 - 5.0*xydprime[...,0]**2 / Rprime**2)
    A7      = (1.0 - 7.0*xyd[...,0]**2 / R**2)
    
    B5 = (1-5. * xyd[...,1]**2 / R**2)
    B7 = (1-7. * xyd[...,1]**2 / R**2)
    
    C5 = (1-5. * xyd[...,2]**2 / R**2)
    C7 = (1-7. * xyd[...,2]**2 / R**2)
   
    d      = xyd[...,2]
    dprime = xydprime[...,2] 
    c      = -self.xyz[2]*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0]*xyd[...,1] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0]**2*xyd[...,1]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    
    K1 =-xyd[...,1] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  
    K3 =-3.*xyd[...,0]*d/R**5 - K2
    
    V      = s - 5.*xyd[...,1]*p*q / R**2
    Vprime = sprime - 5.*xydprime[...,1]*pprime*qprime / Rprime**2
    Vpri   = t + 5.0*d*p*q/R**2
    Vpriprime = tprime + 5*dprime*pprime*qprime / Rprime**2
    
    a1      = (1.5 * (1 - alpha) * xyd[...,0] / R**5)[...,_]
    a1prime = (1.5 * (1 - alpha) * xydprime[...,0] / Rprime**5)[...,_]
    a2      = (1.5 * alpha*p*q / R**5)[...,_]
    a2prime = (1.5 * alpha*pprime*qprime / Rprime**5)[...,_]
    a3      = (3.*p*q / R**5)[...,_]
    a4      = ((1-alpha)*self.sind*self.cosd / alpha)[...,_]
    a5      = ((1-alpha)*3. / R**5)[...,_]
    a6      = (alpha*15.*c / R**7)[...,_]
    
    b1      = (0.5 * (1-alpha) / R**3 )[...,_]
    b1prime = (0.5 * (1-alpha) / Rprime**3)[...,_]
    b2      = (1.5 * alpha*V / R**5)[...,_]
    b2prime = (1.5 * alpha*Vprime / Rprime**5)[...,_]
    b3      = (3.*V / R**5)[...,_]
    b6      = (alpha*3.*c / R**5)[...,_]
    
    c2      = (1.5*alpha*Vpri / R**5)[...,_]
    c2prime = (1.5*alpha*Vpriprime / Rprime**5)[...,_]
    c3      = (3*Vpri / R**5)[...,_]
    
    d1      = (( 1-alpha ) / R**3)[...,_]
    
    duAdx = a1 * numpy.array( [ numpy.zeros_like(s) , -s, t ] ).T \
          + a2 * numpy.array( [ A5 , -5*xyd[...,0]*xyd[...,1]/R**2 , -5*xyd[...,0]*d/R**2 ] ).T
    duAdxprime = a1prime * numpy.array( [ numpy.zeros_like(sprime) , -sprime, tprime ] ).T \
               + a2prime * numpy.array( [ A5prime , -5*xydprime[...,0]*xydprime[...,1]/Rprime**2 , \
                                           -5*xydprime[...,0]*dprime/Rprime**2 ] ).T          
    duBdx = a3 * numpy.array( [ -A5 , 5*xyd[...,0]*xyd[...,1]/R**2 , 5*c*xyd[...,0]/R**2 ] ).T \
          + a4 * numpy.array( [ J3, J1, K3] ).T       
          
    duCdx = a5 * numpy.array( [ t*A5 , xyd[...,0] * (self.cos2d - 5*xyd[...,1]*t/R**2) , \
                               xyd[...,0] * (2+A5) * self.sind*self.cosd ] ).T \
          - a6 * numpy.array( [ p*q*A7 , xyd[...,0] * (s - 7*xyd[...,1]*p*q / R**2) , \
                                 xyd[...,0] * (t + 7*d*p*q/R**2)] ).T


    duAdy = b1 * numpy.array( [ numpy.zeros_like(s) , self.sin2d - 3.*xyd[...,1]*s / R**2 , \
                               -(self.cos2d - 3.*xyd[...,1]*t / R**2) ]).T \
          + b2 * xyd \
          + a2 * [ 0, 1, 0] 
    duAdyprime = b1prime * numpy.array( [ numpy.zeros_like(sprime) , self.sin2d - 3.*xydprime[...,1]*sprime / Rprime**2 , \
                                          -(self.cos2d - 3.*xydprime[...,1]*tprime / Rprime**2) ]).T \
               + b2prime * xydprime \
               + a2prime * [ 0, 1, 0] 
    duBdy =-b3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c] ).T \
          - a3 * [ 0, 1, 0 ]  \
          + a4 * numpy.array( [ J1, J2, K1 ] ).T
    duCdy = a5 * numpy.array( [ xyd[...,0] * (self.cos2d - 5*xyd[...,1]*t / R**2) , \
                                 2*xyd[...,1]*self.cos2d + t*B5 , \
                                 xyd[...,1]*A5*self.sind*self.cosd ] ).T \
          + b6 * numpy.array( [-(5*xyd[...,0]/R**2) * (s - 7.*xyd[...,1]*p*q / R**2) , \
                                 self.sin2d - 10.*xyd[...,1]*s / R**2 - 5.*p*q*B7 / R**2 , \
                                -((3+A5) * self.cos2d + 35.*xyd[...,1]*d*p*q / R**4) ] ).T




    duAdz = b1 * numpy.array( [ numpy.zeros_like(s) , self.cos2d + 3*d*s/R**2 , self.sin2d - 3*d*t/R**2 ] ).T\
          + c2 * xyd \
          - a2 * [ 0, 0, 1 ]
    duAdzprime = b1prime * numpy.array( [ numpy.zeros_like(sprime) , self.cos2d + 3*dprime*sprime/Rprime**2 , 
                                           self.sin2d - 3*dprime*tprime/Rprime**2 ] ).T \
               + c2prime * xydprime \
               - a2prime * [ 0, 0, 1 ]
    duBdz =-c3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
          + a4 * numpy.array( [ -K3 , -K1 , A3/R**3 ] ).T
    duCdz =-a5 * numpy.array( [ xyd[...,0] * (self.sin2d - 5*d*t/R**2) , \
                                 d*B5*self.cos2d + xyd[...,1]*C5*self.sin2d, \
                                 d*A5*self.sind*self.cosd ] ).T \
          - b6 * numpy.array( [ (5*xyd[...,0]/R**2) * (t + 7*d*p*q/R**2) , \
                                 (3+A5) * self.cos2d + 35.*xyd[...,1]*d*p*q/R**4 , \
                                 self.sin2d - 10*d*t/R**2 + 5*p*q*C7/R**2 ] ).T 
                                                                                                                                                                                                                                                            
    uC =  d1 * numpy.array( [3*xyd[...,0]*t / R**2, \
                            -(self.cos2d - 3*xyd[...,1]*t / R**2), \
                            -A3*self.sind *self.cosd] ).T \
       +  b6 * numpy.array( [-5*xyd[...,0]*p*q/R**2 , s-5*xyd[...,1]*p*q/R**2 , t+5*d*p*q/R**2] ).T
    
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz + uC
 
    G =  numpy.array( [ dudx.T, dudy.T, dudz.T ]).T.swapaxes(-2,-1)
    return numpy.dot( numpy.dot( self.rotmat, G ), self.rotmat.T ).swapaxes(0,1)
