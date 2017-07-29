import numpy
from source import Source, diag

_ = numpy.newaxis
PI = numpy.pi

class StrikeSource( Source ):
  def __init__( self, xyz, delta=90, azimuth=0 ):
    self.xyz = numpy.asarray( xyz )
    assert self.xyz.ndim == 1 and self.xyz.size == 3
    
    self.delta = delta * PI / 180        
    self.azimuth   = azimuth * PI / 180
    self.cosd  = numpy.cos(self.delta)
    self.sind  = numpy.sin(self.delta)
    self.rotmat = numpy.array( [ [numpy.cos(self.azimuth),numpy.sin(self.azimuth), 0] , [-numpy.sin(self.azimuth),numpy.cos(self.azimuth),0],[0,0,1] ] )

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

    a1      = (0.5 * ( 1 - alpha ) / R**3)[...,_]
    a1prime = (0.5 * ( 1 - alpha ) / Rprime**3)[...,_]
    a2      = (1.5 * ( alpha*xyd[...,0]*q ) / R**5)[...,_]
    a2prime = (1.5 * ( alpha*xydprime[...,0]*qprime ) / Rprime**5)[...,_]
    a3      = (( 3.0*xyd[...,0]*q ) / R**5)[...,_]
    a4      = (( 1 - alpha )*self.sind / alpha)[...,_]
    a5      = (( 1 - alpha ) / R**5 )[...,_]
    a6      = (alpha*3.0*c / R**5 )[...,_]

    I1 =  xyd[...,1] * ( ( 1. / (R * (R+xyd[...,2])**2) ) - xyd[...,0]**2*( (3.0*R+xyd[...,2]) / (R**3 * (R+xyd[...,2])**3) ) )
    I2 =  xyd[...,0] * ( ( 1. / (R*(R+xyd[...,2])**2) ) - xyd[...,1]**2*( (3.0*R+xyd[...,2])/(R**3 *(R+xyd[...,2])**3) ) )
    I4 = -xyd[...,0] * xyd[...,1] * ( 2*R+xyd[...,2] ) / ( R**3 * (R+xyd[...,2])**2 )
          
    uA      =  a1 * numpy.array( [ q , xyd[...,0]*self.sind , -xyd[...,0]*self.cosd ] ).T \
            +  a2 * xyd
    uAprime =  a1prime * numpy.array( [ qprime , xydprime[...,0]*self.sind , -xydprime[...,0]*self.cosd ] ).T \
            +  a2prime * xydprime
    uB = -a3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
       -  a4 * numpy.array( [ I1, I2, I4 ] ).T   
    uC =  a5 * numpy.array( [ -( R**2 - 3.0 * xyd[...,0]**2 ) * self.cosd ,      \
                                3.0 * xyd[...,0] * xyd[...,1] * self.cosd ,   \
                               -3.0 * xyd[...,0] * xyd[...,1] * self.sind ] ).T \
       +  a6 * numpy.array( [ q * ( 1 - (5 * xyd[...,0]**2 / R**2) ) , \
                               xyd[...,0] * ( self.sind - (5 * xyd[...,1] * q / R**2) ), \
                               xyd[...,0] * ( self.cosd + (5 * xyd[...,2] * q / R**2) ) ] ).T
    
    return numpy.dot( uA - uAprime + uB + xyz[...,2,_]*uC, self.rotmat.T)
    

  def gradient( self, xyz, poisson=0.25 ):
    xyz = numpy.asarray( xyz )
    
    xyd      = numpy.dot( xyz * [1,1,-1] - self.xyz, self.rotmat )
    xydprime = numpy.dot( xyz - self.xyz, self.rotmat )
    

    alpha  = .5 / ( 1 - poisson )
    R      = numpy.linalg.norm( xyd, axis=-1 )
    Rprime = numpy.linalg.norm( xydprime, axis=-1 )
    q      = (xyd[...,1]*self.sind - xyd[...,2]*self.cosd)
    qprime = (xydprime[...,1]*self.sind - xydprime[...,2]*self.cosd)
    
    A3      = 1.0 - 3.0*xyd[...,0]**2 / R**2
    A3prime = 1.0 - 3.0*xydprime[...,0]**2 / Rprime**2
    A5      = 1.0 - 5.0*xyd[...,0]**2 / R**2
    A5prime = 1.0 - 5.0*xydprime[...,0]**2 / Rprime**2
    A7      = 1.0 - 7.0*xyd[...,0]**2 / R**2
    
    B5 = 1-5. * xyd[...,1]**2 / R**2
    B7 = 1-7. * xyd[...,1]**2 / R**2
    
    C7 = 1-7. * xyd[...,2]**2 / R**2
   
    d = xyd[...,2]
    c = -self.xyz[2]*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0]*xyd[...,1] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0]**2*xyd[...,1]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J4 =-3.0*xyd[...,0]*xyd[...,1] / R**5 - J1
    
    K1 =-xyd[...,1] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  

    U         = self.sind - 5*xyd[...,1]*q / R**2
    Uprime    = self.sind - 5*xydprime[...,1]*qprime / Rprime**2
    Upri      = self.cosd + 5*d*q / R**2
    Upriprime = self.cosd + 5*xydprime[...,2]*qprime / Rprime**2
    
    a1      = (0.5 * (1 - alpha) / R**3)[...,_]
    a1prime = (0.5 * (1 - alpha) / Rprime**3)[...,_]
    a2      = (1.5 * (alpha * q) / R**5)[...,_]
    a2prime = (1.5 * (alpha * qprime) / Rprime**5)[...,_]
    a3      = ((3.0 * q) / R**5)[...,_]
    a4      = ((1 - alpha) * self.sind / alpha)[...,_]
    a5      = (3 * (1 - alpha) / R**5)[...,_]
    a6      = (alpha*3.0*c / R**5)[...,_]
    b2      = (1.5* alpha*xyd[...,0]*U / R**5)[...,_]
    b2prime = (1.5* alpha*xydprime[...,0]*Uprime / Rprime**5)[...,_]
    b3      = (3*xyd[...,0]*U / R**5)[...,_]
    c2      = (1.5*alpha*xyd[...,0]*Upri / R**5)[...,_]
    c2prime = (1.5* alpha*xydprime[...,0]*Upriprime / Rprime**5)[...,_]
    c3      = (3*xyd[...,0]*Upri / R**5)[...,_]
          
    duAdx      = a1 * numpy.array( [ -3.0*xyd[...,0]*q / R**2 , A3*self.sind , -A3*self.cosd ] ).T \
               + a2 * xyd * ( A5[...,_] + [1,0,0] )
    duAdxprime = a1prime * numpy.array( [ -3.0*xydprime[...,0]*qprime / Rprime**2 , A3prime*self.sind , -A3prime*self.cosd ] ).T \
               + a2prime * xydprime * ( A5prime[...,_] + [1,0,0] )
          
    duBdx =-a3 * ( A5[...,_] + [1,0,0] ) * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
          - a4 * numpy.array( [ J1, J2, K1 ] ).T
    duCdx = a5 * ( A5[...,_] + [2,0,0] ) * numpy.array( [ xyd[...,0]*self.cosd , xyd[...,1]*self.cosd , -xyd[...,1]*self.sind ] ).T \
          + a6 * numpy.array( [ -5*xyd[...,0]*q*(2+A7) / R**2 , A5*self.sind - 5*xyd[...,1]*q*A7 / R**2 , A5*self.cosd + 5*d*q*A7 / R**2 ] ).T
    duAdy = a1 * numpy.array( [ self.sind - 3.0*xyd[...,1]*q / R**2 , -3.0*xyd[...,0]*xyd[...,1]*self.sind / R**2 , 3.0*xyd[...,0]*xyd[...,1]*self.cosd / R**2 ] ).T \
          + b2 * xyd \
          + a2 * [0,1,0] * xyd[...,0,_]

    duAdyprime = a1prime * numpy.array( [ self.sind - 3.0*xydprime[...,1]*qprime / Rprime**2 ,      \
                                  -3.0*xydprime[...,0]*xydprime[...,1]*self.sind / Rprime**2,    \
                                          3.0*xydprime[...,0]*xydprime[...,1]*self.cosd / Rprime**2 ] ).T \
               + b2prime * xydprime \
               + a2prime * [0,1,0] * xydprime[...,0,_]
    duBdy =-b3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
          - a3 * [0,1,0]* xyd[...,0,_]\
          - a4 * numpy.array( [ J2, J4, K2 ] ).T       
    duCdy = a5 * numpy.array( [ xyd[...,1]*A5*self.cosd , xyd[...,0]*B5*self.cosd , -xyd[...,0]*B5*self.sind ] ).T \
          + a6 * numpy.array( [ A5*self.sind - 5*xyd[...,1]*q*A7 / R**2 ,        \
                        -5*xyd[...,0]*(2*xyd[...,1]*self.sind + q*B7) / R**2, \
                                5*xyd[...,0]*(d*B7*self.sind - xyd[...,1]*C7*self.cosd) / R**2 ] ).T

    duAdz = a1 * numpy.array( [ self.cosd + 3*d*q / R**2 , 3*xyd[...,0]*d*self.sind / R**2 , -3*xyd[...,0]*d*self.cosd / R**2 ] ).T\
          + c2 * xyd \
          - a2 * [0,0,1] * xyd[...,0,_]
    duAdzprime = a1prime * numpy.array( [ self.cosd +3*xydprime[...,2]*qprime / Rprime**2 ,         \
                                          3*xydprime[...,0]*xydprime[...,2]*self.sind / Rprime**2,  \
                                         -3*xydprime[...,0]*xydprime[...,2]*self.cosd / Rprime**2 ]).T\
               + c2prime * xydprime\
               - a2prime * [0,0,1] * xydprime[...,0,_]
    
    duBdz =-c3 * numpy.array( [ xyd[...,0] , xyd[...,1] , c ] ).T \
          + a4 * numpy.array( [ K1 , K2 , 3*xyd[...,0]*xyd[...,1] / R**5 ] ).T 
    duCdz = a5 * numpy.array( [ -d*A5*self.cosd , d*5*xyd[...,0]*xyd[...,1]*self.cosd / R**2 , -d*5*xyd[...,0]*xyd[...,1]*self.sind / R**2 ] ).T \
          + a6 * numpy.array( [ A5*self.cosd+5*d*q*A7 / R**2 ,                          \
                                5*xyd[...,0]*(d*B7*self.sind - xyd[...,1]*C7*self.cosd) / R**2, \
                                5*xyd[...,0]*(2*d*self.cosd - q*C7) / R**2 ] ).T                 
                                                      
    uC =  a5 * numpy.array( [-( R**2/3.0 - xyd[...,0]**2 )*self.cosd , xyd[...,0]*xyd[...,1]*self.cosd , -xyd[...,0]*xyd[...,1]*self.sind ] ).T \
       +  a6 * numpy.array( [ q*( 1 - (5*xyd[...,0]**2 / R**2) ) , xyd[...,0]*( self.sind - (5*xyd[...,1]*q / R**2) ) , xyd[...,0]*(self.cosd + (5*xyd[...,2]*q / R**2)) ] ).T   
    
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz + uC
 
    G = numpy.array( [ dudx.T, dudy.T, dudz.T ]).T.swapaxes(-2,-1)
    return numpy.dot( self.rotmat, numpy.dot( G, self.rotmat.T ) ).swapaxes(0,1)


