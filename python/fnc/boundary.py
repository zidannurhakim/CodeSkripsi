##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import numpy as np
from scipy import signal


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def searchInnerBound(img):

    Y = img.shape[0]
    X = img.shape[1]
    sect = X/4 		
    minrad = 10
    maxrad = sect*0.8
    jump = 4 	

    sz = np.array([np.floor((Y-2*sect)/jump),
                    np.floor((X-2*sect)/jump),
                    np.floor((maxrad-minrad)/jump)]).astype(int)

    # Resolution of the circular integration
    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y*jump
    x = sect + x*jump
    r = minrad + r*jump
    hs = ContourIntegralCircular(img, y, x, r, angs)

    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur
    sm = 3 		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y*jump
    inner_x = sect + x*jump
    inner_r = minrad + (r-1)*jump

    # Integro-Differential operator fine (pixel-level precision)
    integrationprecision = 0.1 		
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump*2),
                          np.arange(jump*2),
                          np.arange(jump*2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Bluring
    sm = 3 		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r


#------------------------------------------------------------------------------
def searchOuterBound(img, inner_y, inner_x, inner_r):
    maxdispl = np.round(inner_r*0.15).astype(int)

    minrad = np.round(inner_r/0.8).astype(int)
    maxrad = np.round(inner_r/0.3).astype(int)

    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    sm = 7 
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r


#------------------------------------------------------------------------------
def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
   
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)

    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1

    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    # Return
    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)

