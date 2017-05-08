import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.spatial.distance import euclidean
from cmath import sqrt
from math import cos, sin, pow, e, pi


def recursive_filter(image, sigma=1.1):
    """
    Implement one-dimensional recursive filter
    """

    # nested functions
    def anticausal_filter(n, mask, x, index):
        """
        n is the index we are looking for
        """

        result = 0

        if n > mask.shape[1] - 5:
            mask[index:index+1, n] = result
            return result

        # calculate first summation
        mul = a_minus * x[n: n + 4]
        first_summation = mul.sum()

        # calculate second summation
        second_summation = 0
        for m, bm in enumerate(b_minus, 1):

            if mask[index:index+1, n] < 0:
                second_summation += bm * anticausal_filter(n + m, mask, x, index)
            else:
                second_summation += bm * mask[index:index+1, n]

        result = first_summation - second_summation

        # update position
        mask[index:index+1, n] = result
        return result


    def causal_filter(n, mask, x, index):
        """
        n is the index we are looking for
        """

        result = 0

        if n < 3:
            mask[index:index+1, n] = result
            return result

        # calculate first summation
        mul = x[n-3: n+1] * a_plus
        first_summation = mul.sum()

        # calculate second summation
        second_summation = 0
        for m, bm in enumerate(b_plus, 1):

            if mask[index:index+1, n] < 0:
                second_summation += bm * causal_filter(n-m, mask, x, index)
            else:
                second_summation += bm * mask[index:index+1, n]

        result = first_summation - second_summation

        # update position
        mask[index:index+1, n] = result
        return result

    # create coefficients
    a_plus, b_plus = create_coefficients1(sigma)
    a_minus, b_minus = create_coefficients2(a_plus, b_plus)

    a_plus = a_plus[::-1]

    n = size_row = image.shape[1]

    # create masks
    causal_mask = image[:] * -1
    anticausal_mask = image[:] * -1

    # apply causal filter
    for i, row in enumerate(image):
        causal_filter(n-1, causal_mask, row, i)

    # apply anti-causal filter
    for i, row in enumerate(image):
        anticausal_filter(1, anticausal_mask, row, i)

    response = (1.0/sigma * sqrt(2.0 * pi)) * (causal_mask + anticausal_mask)

    #msc.imsave("task3_causal_{}.png".format(direction), causal_mask.T)
    #msc.imsave("task3_anti_{}.png".format(direction), anticausal_mask.T)
    #msc.imsave("task3_result_{}.png".format(direction), abs(response.T))

    return(abs(response))


def create_coefficients1(sigma):
    """
    Computation of coefficients a_plus and b_plus
    """

    a1, a2 = 1.68, -0.6803
    b1, b2 = 3.735, -0.2598
    g1, g2 = 1.783, 1.7230
    o1, o2 = 0.6318, 1.997

    a_plus, b_plus = [], []

    # a0
    a_plus.append(a1 + a2)

    # a1
    l = pow(e, -(g2 / sigma)) * (b2 * sin(o2/sigma) - (a2 + 2*a1) * cos(o2/sigma))
    r = pow(e, -(g1 / sigma)) * (b1 * sin(o1/sigma) - (2*a2 + a1) * cos(o1/sigma))
    a_plus.append(l + r)

    # a2
    l = 2 * pow(e, -((g1 + g2) / sigma)) * ((a1 + a2) * cos(o2/sigma)\
            * cos(o1/sigma) - cos(o2/sigma) * b1 * sin(o1/sigma)\
            - cos(o1/sigma) * b2 * sin(o2/sigma))
    r = a2 * pow(e, -2 * (g1 / sigma)) + a1 * pow(e, -2 * (g2 / sigma))
    a_plus.append(l + r)

    # a3
    l = pow(e, -((g2 +  2 * g1) / sigma)) * (b2 * sin(o2/sigma) - a2 * cos(o2/sigma))
    r = pow(e, -((g1 +  2 * g2) / sigma)) * (b1 * sin(o1/sigma) - a1 * cos(o1/sigma))
    a_plus.append(l + r)

    # b1
    b = -2 * pow(e, -(g2/sigma)) * cos(o2/sigma) -2 * pow(e, -(g1/sigma)) * cos(o1/sigma)
    b_plus.append(b)

    # b2
    b = 4 * cos(o2/sigma) * cos(o1/sigma) * pow(e, -(g1 +g2)/sigma)\
            + pow(e, -2 * (g2/sigma)) + pow(e, -2 * (g1/sigma))
    b_plus.append(b)

    # b3
    b = -2 * cos(o1/sigma) * pow(e, -(g1 + 2*g2)/sigma) - 2 *\
            cos(o2/sigma) * pow(e, -(g2 + 2*g1)/sigma)
    b_plus.append(b)

    # b4
    b = pow(e, -(2*g1 + 2*g2)/sigma)
    b_plus.append(b)

    return np.asarray(a_plus), np.asarray(b_plus)


def create_coefficients2(a_plus, b_plus):
    """
    """
    b_minus = b_plus[:]
    a_minus = np.zeros(shape=(4,))

    # a1
    a_minus[0] = a_plus[1] - b_plus[0] * a_plus[0]

    # a2
    a_minus[1] = a_plus[2] - b_plus[1] * a_plus[0]

    # a3
    a_minus[2] = a_plus[3] - b_plus[2] * a_plus[0]

    # a4
    a_minus[3] = - b_plus[3] * a_plus[0]

    return a_minus, b_minus


def readIntensityImage(filename):
    f = msc.imread(filename).astype("float")
    return f


def writeIntensityImage(f, filename):
    msc.toimage(f, cmin=0, cmax=255).save(filename)


def createGaussianFilterBySize( size=(3,3) ):
    sigmas = [((s-1)/2)/2.575 for s in size]
    return createGaussianFilter(sigmax = sigmas[0], sigmay=sigmas[1], size=size)


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
