#!/usr/bin/env python
# coding: utf-8

# ### Finds distance to closest clone for a given Fundamental Domain

__version__ = '1.0'

get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv, pinv
import itertools as it




# ## Distance Functions: Sets up which computation is of interest (parameter search or nearest clone)

def sampleTopol(Manifold, L_Scale, pos, angles):
    
    """
    Overview function that holds the subfunctions to be called in from 'sampleTopology.py' file. To be used when sampling the manifold, 
    rather than interest in a single point. Leads through a computationally simpler process.
    -----------------------------------------
    
    - Calls the constructions function to determine the manifold parameters
    - Calls the E_general_topol function that contains all the relevant functions and returns a single value
    
    -----------------------------------------
    Returns: array of distances from the initial given points to their closest clones
    """
    
    M, translations, pureTranslations, E1Dict, translationList, genNum, x0 = constructions(Manifold, L_Scale, angles)
    
    dist = E_general_topol(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList)
    return (dist)


def distance_to_CC(Manifold, L_Scale, pos, angles):

    """
    Function called from jupyter file
    -----------------------------------------
    
    - Takes the given parameters from the jupyter file input and calls the construction function returning the generators, matricies, and 
    necessary values for the following calculations. 
    
    - Identifies if intersted in the trivial topologies (E1 or E11) and calls the necessary function, skipping many steps if interested in trivial version.
    
    - Runs the respective function that contains all the core functions for the code (E_general or E1_Associataed_Trans)
    
    -----------------------------------------
    Returns: location of closest clone, distance to clone, and combination of generators applied to reach this clone
    """

    M, translations, pureTranslations, E1Dict, translationList, genNum, x0 = constructions(Manifold, L_Scale, angles)

    if (Manifold in {'E1', 'E11'}):
        closestClone, dist, genApplied = E1_Associated_Trans(pureTranslations, pos)

    else:
        closestClone, dist, genApplied = E_general(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList)

    return(closestClone, dist, genApplied)




# ## Code for performing the distance calculations

def constructions(Manifold, L_Scale, angles):
    
    """    
    Constructs the generators from the given parameters from the python file
    -----------------------------------------
    
    
    -Groups if your chosen manifold uses 2 or 3 generators to characterize the fundamental domain.
    -Calls the manifolds class depending on if the chosen topology has 2 or 3 generators. Takes the given parameters and return 
    the generators for that specfic manifold.
    -If origin is placed at the center of the fundamental domain (nearly all should be True), calls the translation function that 
    returns all arangements of translations for the associated E1
    
    -----------------------------------------
    Returns: the initial parameters along with the translation list described above
    
    """
    
    _3Gen = {'E1','E2','E3','E4','E5','E6'}
    _2Gen = {'E11','E12','E16'}
    
    if (Manifold in _3Gen):
        genNum = 3
        M, translations, pureTranslations, E1Dict, center, x0 = manifolds.construct3Generators(Manifold, L_Scale, angles)
        
        if center == True:
            translationList = findAllTranslationsCenter(pureTranslations, genNum)
        else:
            translationList = findAllTranslationsCorner(pureTranslations)
        
    elif (Manifold in _2Gen):
        genNum = 2
        M, translations, pureTranslations, E1Dict, center, x0 = manifolds.construct2Generators(Manifold, L_Scale, angles)
        translationList = findAllTranslationsCenter(pureTranslations, genNum)
        
    return (M, translations, pureTranslations, E1Dict, translationList, genNum, x0)



def E1_Associated_Trans(pureTranslations, pos):
    
    """
    Performs all the calculations for the trivial topologies
    -----------------------------------------
    
    - Labels the generators 
    - Sets all the translations to be pure translations (the side edges of the fundamental domains)
    - Calculates the distance for each of the translation vectors in an array
    - Finds the shortest vector
    - Gets the index of the shortest translation vector
    - Labels 'shortestTrans' as the generator associated with the shortest pure translaiton vector
    
    -----------------------------------------
    Returns: the location of the neareset clone, the distance, and the generator associated with that clone
    """
    
    gens = ['g1','g2','g3']
    
    translationList = pureTranslations
    nearestClone = [distance.euclidean(pureTranslations[i], x0) for i in range(len(pureTranslations))]
    _minNear = min(nearestClone)
    indexOfClone = nearestClone.index(_minNear)
    shortestTrans = gens[indexOfClone]
    
    return(pureTranslations[indexOfClone] + pos, _minNear, shortestTrans)



def E_general(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList):
    
    """
    Function that contains and runs all the core functions of the code for the non-trivial topologies
    -----------------------------------------
    
    - Function that 'finds pattern of clones for associated E1 and generators applied to reach them'
    - Function that 'translates all the clones from the associated E1 to find the full list of clone positions'
    - Function that 'finds the distances to all the clones in the same layer as the initial point'
    - Function that 'finds the closest clone from all layers' and returns [location of the closest clone (x,y,z), distance to closest clone]
    - Function that determines the generator combination that gives the location of the closest clone
    
    -----------------------------------------
    Returns: [xyz coordinates of closest clone, distance to closest clone, generator combination to closest clone]
    """
    
    clonePositions, genApplied = findClones(pos, x0, M, translations, E1Dict, genNum)
    translatedClonePos = [translateClones(clonePositions[i], translationList) for i in range(len(clonePositions))]
    nearestFromLayer = [distances(translatedClonePos[i], pos, x0) for i in range(len(translatedClonePos))]
    closestClone = findClosestClone(nearestFromLayer, pureTranslations, x0, pos)
    generatorCombo = findGeneratorCombo(closestClone[0], clonePositions, pureTranslations, pos, E1Dict, Manifold, genApplied, genNum)
    
    return (closestClone[0], closestClone[1], generatorCombo)


def E_general_topol(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList):
    
    """
    Function that contains and runs all the core functions of the code for the non-trivial topologies (without determining the generator combination for nearest clone)
    -----------------------------------------
    
    - Function that 'finds pattern of clones for associated E1 and generators applied to reach them'
    - Function that 'translates all the clones from the associated E1 to find the full list of clone positions'
    - Function that 'finds the distances to all the clones in the same layer as the initial point'
    - Function that 'finds the closest clone from all layers' and returns [location of the closest clone (x,y,z), distance to closest clone]
    
    -----------------------------------------
    Returns: [xyz coordinates of closest clone]
    """
    
    clonePositions, genApplied = findClones(pos, x0, M, translations, E1Dict, genNum)
    translatedClonePos = [translateClones(clonePositions[i], translationList) for i in range(len(clonePositions))]
    nearestFromLayer = [distances(translatedClonePos[i], pos, x0) for i in range(len(translatedClonePos))]
    closestClone = findClosestClone(nearestFromLayer, pureTranslations, x0, pos)
    
    return (closestClone[1])


# Finds all possible translations in the positive direction by creating combinations of pure translations

def findAllTranslationsCorner(pureTranslations):
    
    """
    Determines all combinations of pure translations for fundamental domains with origin set to be the corner (unused/outdated but for different sampling methods)
    -----------------------------------------
    
    - Makes use of itertools function to find all combinations of possible pure translations arrangements. Has a nested array form for the different arangements
    - Adds the translation vectors in the nested array to create a simple list of all the possible pure translations
    - Appends the location of the farthest (upper) corner fundamental domain to the list of translations (necessary based on the itertools output)
    - Appends the location of the farthest (lower) corner fundamental domain to the list of translations (necessary based on the itertools output)
    - Replaces the first element with the origin (necessary based on the itertools output)
    - Changes the form of the array and element type to be readable as arrays (list)
    - Removes the duplicate pure translation vectors
    
    -----------------------------------------
    Returns: All pure translation vectors in a nested numpy array
    """
    
    _trans1 = [list(it.combinations_with_replacement(pureTranslations, i)) for i in range(len(pureTranslations) + 2)]
    _trans2 = [[(np.add.reduce(_trans1[i][j])) for j in range(len(_trans1[i]))] for i in range(len(_trans1))]
    
    
    _trans2.append([pureTranslations[0] - pureTranslations[1]])
    _trans2.append([pureTranslations[1] - pureTranslations[0]])
    
    _trans2[0] = [[0,0,0]]
    
    transUpPlane = list(it.chain.from_iterable(_trans2))
    allnewTrans = np.array((np.unique(transUpPlane, axis = 0)))
    
    return(allnewTrans)



def findAllTranslationsCenter(pureTranslations, genNum):
   
    """
    Determines all combinations of pure translations for fundamental domains with origin set to be the center. Used as primary method 
    of finding arrangements of associated E1 translations. Generally, manually finds the 26 associated E1s surrounding the base chosen 
    associated E1 by combining the pure translation vectors
    -----------------------------------------
    
    - Manually creating an array of pure translation vectors for all the 8 objects neighboring the initial chosen point
    - Checks the number of generators to determine if we have an additional axis of neighboring clones
    - If 3 generators, combines the array of pure translation vectors in the same plane as the initial point, with the pure translation vectors in the plane 'above' the initial point
    - If 2 generators, does not need a vertical translation list of associated E1s and so just returns the pure translations vectors in the same plane
    
    -----------------------------------------
    Returns: The array of pure translation vectors that produce all the necessary neighboring associated E1s
    """
    
    layerTrans = [pureTranslations[0],    pureTranslations[1],       -pureTranslations[0],   -pureTranslations[1],
               -2*pureTranslations[0], -2*pureTranslations[1],      2*pureTranslations[0],  2*pureTranslations[1],
                  pureTranslations[0] + pureTranslations[1],        pureTranslations[0] - pureTranslations[1], 
                 -pureTranslations[0] + pureTranslations[1],       -pureTranslations[0] - pureTranslations[1]] 
                 
    if genNum == 3:
        allnewTrans = np.concatenate([layerTrans, layerTrans + pureTranslations[2], [pureTranslations[2]]])
        return(allnewTrans)
    else:
        return(layerTrans)


# ### Finds all the clones up to associated E1
# Takes the original position and applies combinations of the generators (up to E1 for each generator) and returns a list of clones


def findClones(pos, x0, M, translations, E1Dict, genNum):
    
    """
    ISSUES: Probably should be rewritten, this applies generators in g1, g2, then g3. Might not need all of them, might to not be in that order etc. 
    
    -----------------------------------------
    Determines the pattern of generators and location of new clones necessary to fill the associated E1
    
    First Loop: Determines the full arangement of generators to fill the associated E1. For example, an element 
    of clonePos may look like [1,1,2] which means applying g1 . g2 . g3 . g3 to the initial position.
    
    Second Loop: Determines the new position of the original point after applying the combination of generators described in the corrosponding element in clonePos. 
    - Loops over all arangements of generators to fill associated E1
    - If all the elements in generator pattern are not 0 set the initial position to x
    - Loop over each element in each arrangement of generators
    - Apply g1, g2, and g3 the number of times described by the associated element in clonePos. (For example, for clonePos[i] = [1,1,2] it will apply to _x: g1 -> g2 -> g3 -> g3.
    - Adds the new position to the list of all the clones
    
    - If there are no arangements of generators necessary (for trivial cases) thenn fullCloneList is empty and skips second loop
    
    -----------------------------------------
    Returns: (for trivial case) original position and empty arangements of generator; (for non-trivial case) new locations of clones in associated E1 and generator pattern
    """
    
    clonePos = []
    
    for i in range(E1Dict[0]):
        for j in range(E1Dict[1]):
            if(genNum == 3):
                for k in range(E1Dict[2]):
                    clonePos.append([i,j,k])
            else:
                clonePos.append([i,j])

    fullCloneList = []
    
    for i in range(len(clonePos)):
        if not (all(x == 0 for x in clonePos[i])):
            _x = pos
            for j in range(len(clonePos[i])):
                for k in range(clonePos[i][j]):
                    _x = generatorPos(_x, x0, M[j], translations[j])
            fullCloneList.append(_x)
    
    if not fullCloneList:
        return(pos, clonePos)
    else:
        return(fullCloneList, clonePos)


# #### Applies a generator to the input point


def generatorPos(x,x0, M, translations):
    
    """
    Application of a generator to an initial point, x
    -----------------------------------------
    
    - Takes the rotation matrix, initial point, translation vector, and origin and calculates the clone position
    
    -----------------------------------------
    Returns: New clone position   
    """
    
    x_out = M.dot(x-x0) + translations + x0
    
    return(x_out)


# #### Translates a clones position for each of the allowed translations



def translateClones(clonePos, translations):
    
    """
    Translates a given position by all the translations given in the nested translation vector
    -----------------------------------------
    
    - Adds a clone position to a list of translation vectors to create an array of the new location of that clone
    
    -----------------------------------------
    Returns: A nested array of how a specific clone position is translated by all the pure translation vectors    
    """
    
    translatedClonePos = [(clonePos + translations[i]) for i in range(len(translations))]
    
    return(translatedClonePos)


# #### Finds the distance between each translated clone and the original position


def distances(clonePos, pos, x0):
    
    """
    Determines distance between initial position and all elements in the clonePos array. In this case, all elements of clonePos are pure translations of the initial point
    -----------------------------------------
    
    - Creates array of distances between initial point and all the clones in the same 'layer' as that point
    - Finds the minimum distance from this array
    - Determines the position of the closest clone associated with this minimum distance
    
    -----------------------------------------
    Returns: Position of closest clone in that layer and the distance to that clone
    
    Determines the distance between the inital position and each of the clones in a single "layer". That is, after applying a generator (or combination
    of generators) and then all the pre-determined pure translations to that clone, which of these new clones is closest to the original position.
    For example, E3 has 3 unique clones (g3, g3^2, and g3^3) and so has 3 unique layers. This function returns the closest clone (to the original position)
    in each of these layers"""
    
    _TransDist = [distance.euclidean(pos, clonePos[i]) for i in range(len(clonePos))]
    min_TransDist = min(_TransDist)
    closestClonePos = clonePos[_TransDist.index(min_TransDist)]
    
    return(closestClonePos, min_TransDist)


# ### Finds the closest clone from all translated clones
# Takes an input of all the translated clones and compares their positions to the original position. Finds the minimum of pure translations. Returns the minimum of these two values


def findClosestClone(generatedClones, pureTrans, x0, pos):
    
    """
    Determines the closest clone from the clones in the initial points layer, and the clones in the upper and lower layers
    -----------------------------------------
    
    - Creates a nested array where each element contains the location of all the clones in the upper layer and the distance to that clone (from the initial position)
    - Finds the closest clone from the upper layer to the original point (by checking the minimum of the distances so long as they are above some small number to avoid the same point)
    - Finds the closest clone from all the clones in same layer as the original point
    
    -----------------------------------------
    Returns: The minimum of the clones between the same layer and the above layer
    """
    
    _TranslateClone = [[(pureTrans[x] + pos) , distance.euclidean(x0,pureTrans[x])] for x in range(len(pureTrans))] 
    
    _closestTranslatedClone = min(_TranslateClone, key = lambda x: x[1] if (x[1] > 10e-12) else np.nan)
    _closestGeneratedClone = min(generatedClones, key = lambda x: x[1] if (x[1]> 10e-12) else np.nan, default = _closestTranslatedClone)

    return(min((_closestGeneratedClone, _closestTranslatedClone), key = lambda x: x[1]))


# #### Finds Combination of Generators to Produce closest Clone
# TLDR: Takes the input of the closest clones position, all the generated clones (up to associated E1), pure translations, etc. and determines what combinatinos of generators will produce that closest clone position.
# 
# Detailed Description: Takes the list of all clones (up to associated E1) and subtracts the closest clone position. This should produce a set of new points (without a relevant physical interpretation) that are linear combinations of the three pure translation vectors. One of those linear combinations will be a set of integers: the one with the corrosponding, non-trivial, generator. Next, this determines which new point is that vector of only integers (and what those integers are), determines what the original clone was, and combines the original clones generator, and the linear combination of translations to return the full list of generators.


def findGeneratorCombo(pos, clones, translations, origPos, E1Dict, Manifold, genApplied, genNum):
    """
    Find the series of generators that produces the closest clone
    -----------------------------------------
    
    - Need to add some more comments here describing this better
    
    -----------------------------------------
    Returns: List of generators producing the closest clone
    
    Method: Determines which clone from the set produced by the findClones() function gave the closest clone. 
    Finds the linear combination of pure translations applied after. Returns a description of non-trivial generator and pure translations.
    Example would be 'Apply g3 once, pure translations of g1, g2^2'
    """
    
    gens = [f'g1', f'g2', f'g3']
    
    if Manifold in {'E1', 'E11'}:
        _x = [clones - pos]
    else:
        _x = clones - pos
    _z = np.insert(_x, 0,[origPos - pos], axis = 0)
    
    if (genNum == 3):
        n_list = np.around([((_z[i]).dot(inv(translations))) for i in range(len(_z))], decimals = 8)
    else:
        n_list = np.around([((_z[i]).dot(pinv(translations))) for i in range(len(_z))], decimals = 8)
        
    
    test = 0
    maxZeroes = 0

    for i in range(len(n_list)):
        count = 0
        isZero = 0
        for j in range(len(n_list[i])):
            if ((n_list[i][j]).is_integer()) == True:
                count +=1
                if (n_list[i][j]) == 0:
                    isZero +=1
                    
        if count == len(n_list[i]):
            if isZero == len(n_list[i]):
                transArray, case = n_list[i], i
                break
            else:
                transArray, case = n_list[i], i
                maxZeroes = isZero
        
    initialClone = [genApplied[case][i]*gens[i] for i in range(len(genApplied[case]))]
    genList = [k for k in initialClone if k] +[f'pure translations: '] + [(gens[i], -transArray[i]) for i in range(len(genApplied[0]))]
        
    return(genList)


# #### Class containing all the manifolds and relevant quantities


class manifolds:
    """
    Class containing the manifold constructions, divided between 2 and 3 generator manifolds
    For 3 generators need to define 3 generators by defining;
    -----------------------------------------------------------------------------------
    M1, M2, M3 : Rotation matricies associated with g1, g2, g3 respectively
    E1_g1, E1_g2, E1_g3 : Order of rotation matricies that return to identity
    TA1, TA2, TB : 3 component numpy array  defining the the translation vectors 
    """
        
    def construct3Generators(Manifold, L_Scale, angles):
        L1, L2, L3 = L_Scale[0], L_Scale[1], L_Scale[2]
        
        M = []
        
        if (Manifold == "E1"):
            M1 = M2 = M3 = np.identity(3)
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 1
            T1 = TA1 = L1 * np.array([1,0,0])
            T2 = TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = TB  = L3 * np.array([np.cos(angles[1])*np.cos(angles[2]), np.cos(angles[1])*np.sin(angles[2]), np.sin(angles[1])])
            center = False
        
        
        elif (Manifold == "E2"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 2
            M1 = M2 = np.identity(3)
            M3 = np.diag([-1,-1,1])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            TB  = np.array([0, 0, L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = np.array([0, 0, 2*L3])
            center = True
            
            
        elif (Manifold == "E3"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 4
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[0,  1,  0],
                                 [-1, 0,  0],
                                 [0,  0,  1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = np.array([0,0,4*L3])
            
            center = True
            
            if (L1 != L2 or angles[0] != np.pi/2):
                raise ValueError("Restrictions on E3: L1=L2 and alpha = pi/2")
        
        
        elif (Manifold == "E4"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 3
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[-1/2,           np.sqrt(3)/2,  0],
                                 [-np.sqrt(3)/2, -1/2,           0],
                                 [0,              0,             1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            T3 = L3 * np.array([0, 0, 3*np.sin(angles[1])])
            T3 = np.array([0,0,3*L3])
            
            center = True
            
            if (L1 != L2):
                raise ValueError("Restrictions on E4: L1=L2")
           
        
        elif (Manifold == "E5"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 6
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[1/2,           np.sqrt(3)/2,   0],
                                 [-np.sqrt(3)/2, 1/2,            0],
                                 [0,              0,             1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            T3 = np.array([0,0,6*L3])
            
            center = True
            
            if (L1 != L2):
                raise ValueError("Restrictions on E5: L1=L2")
                
                
        elif (Manifold == "E6"):
            LCx, LAy, LBz = L_Scale[0], L_Scale[1], L_Scale[2]
            
            E1_g1 = 2
            E1_g2 = 2
            E1_g3 = 2
            
            M1 = np.diag(([1,  -1, -1]))
            M2 = np.diag(([-1,  1, -1]))
            M3 = np.diag(([-1, -1,  1]))
            
            LAx = LCx
            LBy = LAy
            LCz = LBz
            
            TA1 = np.array([LAx, LAy,    0])
            TA2 = np.array([0,   LBy,  LBz])
            TB  = np.array([LCx,   0,  LCz])
            
            T1 = 2*LAx * np.array([1,0,0])
            T2 = 2*LBy * np.array([0,1,0])
            T3 = 2*LCz * np.array([0,0,1])
            
            center = True          
                
        
            
        translations = np.around(np.array([TA1, TA2, TB]), decimals = 5)
        pureTranslations = np.around(np.array([T1, T2, -T3]), decimals = 5)   #Probably an iffy way to solve this problem, needs to figure out if theres a better way to do this
        associatedE1Dict = np.array([E1_g1, E1_g2, E1_g3])
        M = [M1, M2, M3]
        #x0 = 0.5*np.sum(translations, axis =0)
        x0 = np.array([0,0,0.])
        
        return(M, translations, pureTranslations, associatedE1Dict, center, x0)
            


    def construct2Generators(Manifold, L_Scale, angles):
        #L1, L2, L3x, L3y, L3z = L_Scale[0], L_Scale[1], L_Scale[2], L_Scale[3], L_Scale[4]
        L1, L2 = L_Scale[0], L_Scale[1]
        
        if (Manifold == 'E11'):
            E1_g1 = 1
            E1_g2 = 1
            
            M1 = M2 = np.identity(3)
            TA1 = T1 = L1 * np.array([1,0,0])
            TA2 = T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            
            center = False
            
        
        elif (Manifold == 'E12'):
            E1_g1 = 1
            E1_g2 = 2
            
            M1 = np.identity(3)
            M2 = np.diag([-1, 1, -1])
            
            TA1 = T1 = L1 * np.array([np.cos(angles[0]),0, np.sin(angles[0])])
            TA2 = np.array([0,L2,0])
            
            T2 = np.array([0,2*L2, 0])
            
            center = True
            
            
        translations = np.around(np.array([TA1, TA2]), decimals = 5)
        pureTranslations = np.around(np.array([T1, T2]), decimals = 5)
        associatedE1Dict = np.array([E1_g1, E1_g2])
        M = [M1, M2]
        x0 = np.array([0,0,0.])
        
        return(M,translations, pureTranslations, associatedE1Dict, center, x0)
