__version__ = '1.0'

import SingleCloneDistance as d
import numpy as np

def scatterPoints(randomPoints, translations, precision, center, genNum, L3):
    """
    Creates an array of random points in the given 'fundamental domain'
    -----------------------------------------
    
    - Creates an array of random points by taking random point along each basis vector of the FD and adding them together
    - Does same for 2 generators (where the 3rd vector w/-0.5 and 2*L3 says how far into the infinite dimension we will sample)
    
    -----------------------------------------
    Returns: Scatter of random points within the fundamental domain
    """
    if genNum == 3:
        L_scatter = [(randomPoints[i][0] * translations[0] + randomPoints[i][1] * translations[1] - randomPoints[i][2] * translations[2]) for i in range(precision)]
    else:
        L_scatter = [(randomPoints[i][0] * translations[0] + randomPoints[i][1] * translations[1] + (randomPoints[i][2]-0.5)*np.array([0,0,2*L3])) for i in range(precision)]
    return(L_scatter)


def samplePoints(Manifold, angles, precision, L_Scale):
    
    """
    Repeats everything described in "SingleCloneDist.py" without finding the generator pattern to reach nearest clone. 
    -----------------------------------------
    
    - Creates array of random points of [num of points to test in a chosen manifold, 3]
    - Constructs generators for chosen manifold
    - Runs function to determine cartesian position of points in the manifold
    - Relocates the origin to be in the center of the FD
    
    Precision Loop:
    - Loops over all points being sampled in the manifold
    - Applies the function sampleTopol located in 'SingleCloneDistance.py' file which returns the distance for each point in the given array
    - Tests if point produces circles in the sky (dist<1) and sorts point into respective array
    
    - Determines the fraction (percent) of points that produce circles from all sampled points
    - Sorts into exportable arrays to be used later for plotting/use with pickle
    
    -----------------------------------------
    Returns: [Fraction of points that produce circles, {x of excluded pts, y .., z ..}, {x of allowed pts, y .., z ..}]
    """
    
    if Manifold in {'E1','E2','E3','E4','E5','E6'}:
        genNum = 3
    else:
        genNum = 2
    np.random.seed(1234)
    randomPoints = np.random.rand(precision, 3)
    
    if genNum == 3:
        M, translations, pureTranslations, E1Dict, center, x0 = d.manifolds.construct3Generators(Manifold, L_Scale, angles)
        L_scatter = scatterPoints(randomPoints, pureTranslations, precision, center, genNum, L_Scale[2])
        pos = L_scatter - 0.5*((pureTranslations[0] + pureTranslations[1] - pureTranslations[2]))
    else:
        M, translations, pureTranslations, E1Dict, center, x0 = d.manifolds.construct2Generators(Manifold, L_Scale, angles)
        L_scatter = scatterPoints(randomPoints, pureTranslations, precision, center, genNum, L_Scale[2])
        pos = L_scatter - 0.5*((pureTranslations[0] + pureTranslations[1]))
    
    count = 0
    excludedPoints = []
    allowedPoints = []
    
    

    for k in range(precision):
        dist = d.sampleTopol(Manifold, L_Scale, pos[k], angles)
        # print(dist)
        if dist < 1:
            count +=1
            excludedPoints.append(pos[k])
        else:
            allowedPoints.append(pos[k])

    percents = 1 - (count/precision)
    
    excludedPoints = np.array(excludedPoints).tolist()
    allowedPoints = np.array(allowedPoints).tolist()

    L_x = [allowedPoints[i][0] for i in range(len(allowedPoints))]
    L_y = [allowedPoints[i][1] for i in range(len(allowedPoints))]
    L_z = [allowedPoints[i][2] for i in range(len(allowedPoints))]
    excludedPoints_x = [excludedPoints[i][0] for i in range(len(excludedPoints))]
    excludedPoints_y = [excludedPoints[i][1] for i in range(len(excludedPoints))]
    excludedPoints_z = [excludedPoints[i][2] for i in range(len(excludedPoints))]
            
    return(percents, [excludedPoints_x, excludedPoints_y, excludedPoints_z], [L_x, L_y, L_z])





