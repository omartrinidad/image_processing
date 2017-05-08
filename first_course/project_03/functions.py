import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.spatial.distance import euclidean
from cmath import sqrt

def readIntensityImage(filename):
    f = msc.imread(filename).astype("float")
    return f


def writeIntensityImage(f, filename):
    msc.toimage(f, cmin=0, cmax=255).save(filename)


def createGaussianFilterBySize( size=(3,3) ):
    sigmas = [((s-1)/2)/2.575 for s in size]
    return createGaussianFilter(sigmax = sigmas[0], sigmay=sigmas[1], size=size)
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def createGaussianFilter( sigma=1.0, sigmax=None, sigmay=None, size=None ):
    """
    Author: Minato
    Generate a filter given the sigma
    sigma, sigmax, sigmay all default to 1.0
    default would be a square filter
    also returns filter for each dimension
    """
    sigmax = sigmax or sigma
    sigmay = sigmay or sigma

    size = size or (int( np.ceil( sigmax * 2.575 ) * 2 + 1 ), int( np.ceil( sigmay * 2.575 ) * 2 + 1 ))
    
    msize = size[0]
    x = np.arange( msize )
    gx = np.exp( -0.5 * ( ( x-msize/2) / sigmax ) ** 2 )
    gx /= gx.sum()

    nsize = size[1]
    y = np.arange( nsize )
    gy = np.exp( -0.5 * ( ( y-nsize / 2 ) / sigmay ) ** 2 )
    gy /= gy.sum()

    G = np.outer( gx, gy )
    G /= G.sum()
    return G, gx, gy

def pad2d( mat, pad=1, mode="constant" ):
    """
    Pads Array with defined pading.
    only works with 1d and 2d 
    """
    new_mat = np.array(mat)
    if(new_mat.ndim > 2):
        print('I don\'t do more that 2d.. Sorry.')
        return mat
    if( type(pad) == type(1) ):
        return np.pad(new_mat, pad, mode=mode)
    pad = np.asarray(pad)
    for i in range(len(pad)):
        new_mat = np.array([ np.pad( new_mat[j,:], pad[i], mode=mode )  for j in range(new_mat.shape[0]) ]).T
    # new_mat = np.array([ np.pad( new_mat[j,:], pad[1], mode='constant' )  for j in range(new_mat.shape[0])]).T
    return new_mat

def naiveConvolve2d( img, filter2d ):
    """
    works only with square filters for now
    """
    w = img.shape[0]
    h = img.shape[1]
    m = filter2d.shape[0]
    n = filter2d.shape[1]
    boundm = np.floor( m / 2 )
    boundn = np.floor( n / 2 )
    new_image = np.ndarray( ( w, h ) )
    for x in range( 0, w ):
        for y in range( 0, h ):
            summe = 0.0
            for i in range( 0, m ):
                for j in range( 0, n ):
                    xdash = int( x + ( i - boundm ) )
                    ydash = int( y + ( j - boundn ) )
                    if( 0 > xdash or 
                        w <= xdash or
                        0 > ydash or
                        h <= ydash ):
                        summe += 0.0
                    else:
                        summe += img[ xdash, ydash ] * filter2d[ i, j ]
            new_image[ x, y ] = summe
    return new_image


def discreteDerivative(mat):
    """
    Please send 2d stuff
    """
    w = mat.shape[0]
    h = mat.shape[1]
    # new_mat = np.array((w,h,2) dtype='float')
    padded_mat = pad2d(mat, (1,1), mode="constant")
    new_mat = np.ndarray((w,h,2), dtype='float')
    for x in range(w):
        for y in range(h):
            new_mat[x,y] = [ 
                            ( padded_mat[ x + 1, y + 1 ] - padded_mat[ x, y + 1 ] ) / 2. , 
                            ( padded_mat[ x + 1, y + 1 ] - padded_mat[ x + 1, y ] ) / 2.
                           ]
    return new_mat

def naiveConvolve2dOptimized( img, filter2d ):
    """
    works only with square filters for now
    """
    w = img.shape[0]
    h = img.shape[1]
    m = filter2d.shape[0]
    n = filter2d.shape[1]
    boundm = int(np.floor( m / 2 ))
    boundn = int(np.floor( n / 2 ))
    padded_image = pad2d(img, (boundm, boundn) )
    new_image = np.ndarray( ( w, h ) )    
    for x in range( 0, w ):
        for y in range( 0, h ):
            summe = 0.0
            extract = padded_image[x:(m+x), y:(n+y)]
            mul = np.multiply(extract, filter2d)
            summe = mul.sum()
            new_image[x,y] = summe
    return new_image

def extendFilter(fil, shape):
    w = shape[0]
    h = shape[1]
    m = fil.shape[0]
    n = fil.shape[1]
    px, remx  = divmod(w, m)
    py, remy  = divmod(h, n)
    repx = int(np.ceil(w/float(m)))
    repy = int(np.ceil(h/float(n)))
    new_fil = np.zeros(shape)
    print(repx,repy)
    # print np.tile(fil, (repx, repy)),shape
    new_fil = np.tile(fil, (repx, repy))[0:w,0:h]
    return new_fil

def convolve2dFFT( img, size ):
    sigmas = [((s-1)/2)/2.575 for s in size]
    G,_,_ = createGaussianFilter(sigmax = sigmas[0], sigmay=sigmas[1], size=img.shape)
    Fg = fft.fft2(G)
    Fi = fft.fft2(img)
    pm = np.multiply(Fg,Fi)
    ipm = np.abs(fft.ifftshift(fft.ifft2(pm)))
    return ipm

def convolve1d( img, filter1d ):
    """
    works only with square filters for now
    """
    w = img.shape[0]
    h = img.shape[1]
    m = len( filter1d )
    bound = np.floor( m / 2 )
    new_image = np.ndarray( ( w, h ) )
    for y in range( 0, h ):
        for x in range( 0, w ):
            summe = 0.0
            for i in range( 0, m ):
                xdash = int( x + ( i - bound ) )
                if( 0 > xdash or w <= xdash ):
                    summe += 0.0
                else:
                    summe += img[ xdash, y ] * filter1d[ i ]
            new_image[ x, y ] = summe
    return new_image

def convolve1dOptimized( img, filter1d ):
    """
    works only with square filters for now
    """
    w = img.shape[0]
    h = img.shape[1]
    m = len( filter1d )
    bound = int(np.floor( m / 2 ))
    padded_img = pad2d( img, (bound,) )
    new_img = np.ndarray( ( w, h ) )
    for y in range( 0, h ):
        for x in range( 0, w ):
            summe = 0.0
            extract = padded_img[x:(m+x), y]
            mul = np.multiply(extract, filter1d)
            summe = mul.sum()
            new_img[x,y] = summe
    return new_img

def compare_mat(mat1,mat2):
    return(np.abs(mat1-mat2).sum())

def gaussianFilter(image, size=(3,3)):
    """
    """
    filtered = None
    return filtered


def recursive_filter():
    """
    """
    pass

