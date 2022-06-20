##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import radon


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def findline(img):

    # Pre-processing
    I2, orient = canny(img, 2, 0, 1)
    I3 = adjgamma(I2, 1.9)
    I4 = nonmaxsup(I3, orient, 1.5)
    edgeimage = hysthresh(I4, 0.2, 0.15)

    theta = np.arange(180)
    R = radon(edgeimage, theta, circle=False)
    sz = R.shape[0] // 2
    xp = np.arange(-sz, sz+1, 1)

    # Find for the strongest edge
    maxv = np.max(R)
    if maxv > 25:
        i = np.where(R.ravel() == maxv)
        i = i[0]
    else:
        return np.array([])

    R_vect = R.ravel()
    ind = np.argsort(-R_vect[i])
    u = i.shape[0]
    k = i[ind[0: u]]
    y, x = np.unravel_index(k, R.shape)
    t = -theta[x] * np.pi / 180
    r = xp[y]

    lines = np.vstack([np.cos(t), np.sin(t), -r]).transpose()
    cx = img.shape[1] / 2 - 1
    cy = img.shape[0] / 2 - 1
    lines[:, 2] = lines[:,2] - lines[:,0]*cx - lines[:,1]*cy
    return lines


# ------------------------------------------------------------------------------
def linecoords(lines, imsize):
    
    xd = np.arange(imsize[1])
    yd = (-lines[0,2] - lines[0,0] * xd) / lines[0,1]

    coords = np.where(yd >= imsize[0])
    coords = coords[0]
    yd[coords] = imsize[0]-1
    coords = np.where(yd < 0)
    coords = coords[0]
    yd[coords] = 0

    x = xd
    y = yd
    return x, y


# ------------------------------------------------------------------------------
def canny(im, sigma, vert, horz):
   
    def fspecial_gaussian(shape=(3, 3), sig=1):
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        f = np.exp(-(x * x + y * y) / (2 * sig * sig))
        f[f < np.finfo(f.dtype).eps * f.max()] = 0
        sum_f = f.sum()
        if sum_f != 0:
            f /= sum_f
        return f

    hsize = [6 * sigma + 1, 6 * sigma + 1]  # The filter size
    gaussian = fspecial_gaussian(hsize, sigma)
    im = convolve(im, gaussian, mode='constant')  # Smoothed image
    rows, cols = im.shape

    h = np.concatenate([im[:, 1:cols], np.zeros([rows,1])], axis=1) - \
        np.concatenate([np.zeros([rows, 1]), im[:, 0: cols - 1]], axis=1)

    v = np.concatenate([im[1: rows, :], np.zeros([1, cols])], axis=0) - \
        np.concatenate([np.zeros([1, cols]), im[0: rows - 1, :]], axis=0)

    d11 = np.concatenate([im[1:rows, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d11 = np.concatenate([d11, np.zeros([1, cols])], axis=0)
    d12 = np.concatenate([np.zeros([rows-1, 1]), im[0:rows - 1, 0:cols - 1]], axis=1)
    d12 = np.concatenate([np.zeros([1, cols]), d12], axis=0)
    d1 = d11 - d12

    d21 = np.concatenate([im[0:rows - 1, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d21 = np.concatenate([np.zeros([1, cols]), d21], axis=0)
    d22 = np.concatenate([np.zeros([rows - 1, 1]), im[1:rows, 0:cols - 1]], axis=1)
    d22 = np.concatenate([d22, np.zeros([1, cols])], axis=0)
    d2 = d21 - d22

    X = (h + (d1 + d2) / 2) * vert
    Y = (v + (d1 - d2) / 2) * horz

    gradient = np.sqrt(X * X + Y * Y)  

    orient = np.arctan2(-Y, X)  # Angles -pi to +pi
    neg = orient < 0  # Map angles to 0-pi
    orient = orient * ~neg + (orient + np.pi) * neg
    orient = orient * 180 / np.pi  # Convert to degrees

    return gradient, orient


# ------------------------------------------------------------------------------
def adjgamma(im, g):
    
    newim = im
    newim = newim - np.min(newim)
    newim = newim / np.max(newim)
    newim = newim ** (1 / g)  # Apply gamma function
    return newim


# ------------------------------------------------------------------------------
def nonmaxsup(in_img, orient, radius):
    
    # Preallocate memory for output image for speed
    rows, cols = in_img.shape
    im_out = np.zeros([rows, cols])
    iradius = np.ceil(radius).astype(int)

    # Pre-calculate x and y offsets relative to centre pixel for each orientation angle
    angle = np.arange(181) * np.pi / 180  # Angles in 1 degree increments (in radians)
    xoff = radius * np.cos(angle)  # x and y offset of points at specified radius and angle
    yoff = radius * np.sin(angle)  # from each reference position

    hfrac = xoff - np.floor(xoff)  # Fractional offset of xoff relative to integer location
    vfrac = yoff - np.floor(yoff)  # Fractional offset of yoff relative to integer location

    orient = np.fix(orient)

    # Now run through the image interpolating grey values on each side
    # of the centre pixel to be used for the non-maximal suppression
    col, row = np.meshgrid(np.arange(iradius, cols - iradius),
                           np.arange(iradius, rows - iradius))

    # Index into precomputed arrays
    ori = orient[row, col].astype(int)

    # x, y location on one side of the point in question
    x = col + xoff[ori]
    y = row - yoff[ori]

    # Get integer pixel locations that surround location x,y
    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    # Value at integer pixel locations
    tl = in_img[fy, fx]  # top left
    tr = in_img[fy, cx]  # top right
    bl = in_img[cy, fx]  # bottom left
    br = in_img[cy, cx]  # bottom right

    # Bi-linear interpolation to estimate value at x,y
    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # Check the value on the other side
    map_candidate_region = in_img[row, col] > v1

    x = col - xoff[ori]
    y = row + yoff[ori]

    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    tl = in_img[fy, fx]
    tr = in_img[fy, cx]
    bl = in_img[cy, fx]
    br = in_img[cy, cx]

    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # Local maximum
    map_active = in_img[row, col] > v2
    map_active = map_active * map_candidate_region
    im_out[row, col] = in_img[row, col] * map_active

    return im_out


# ------------------------------------------------------------------------------
def hysthresh(im, T1, T2):
  
    # Pre-compute some values for speed and convenience
    rows, cols = im.shape
    rc = rows * cols
    rcmr = rc - rows
    rp1 = rows + 1

    bw = im.ravel()  # Make image into a column vector
    pix = np.where(bw > T1) # Find indices of all pixels with value > T1
    pix = pix[0]
    npix = pix.size         # Find the number of pixels with value > T1

    # Create a stack array (that should never overflow)
    stack = np.zeros(rows * cols)
    stack[0:npix] = pix         # Put all the edge points on the stack
    stp = npix  # set stack pointer
    for k in range(npix):
        bw[pix[k]] = -1         # Mark points as edges

    O = np.array([-1, 1, -rows - 1, -rows, -rows + 1, rows - 1, rows, rows + 1])

    while stp != 0:  # While the stack is not empty
        v = int(stack[stp-1])  # Pop next index off the stack
        stp -= 1

        if rp1 < v < rcmr:  
            index = O + v  # Calculate indices of points around this pixel.
            for l in range(8):
                ind = index[l]
                if bw[ind] > T2:  # if value > T2,
                    stp += 1  # push index onto the stack.
                    stack[stp-1] = ind
                    bw[ind] = -1  # mark this as an edge point

    bw = (bw == -1)  # Finally zero out anything that was not an edge
    bw = np.reshape(bw, [rows, cols])  # Reshape the image
    return bw

