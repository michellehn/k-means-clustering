# a6.py
# Michelle Nelson (mhn29)
# 11/20/14
"""Classes to perform KMeans Clustering"""

import math
import random
import numpy

# HELPER FUNCTIONS FOR ASSERTS GO HERE
def is_point(thelist):
    """Return: True if thelist is a list of int or float"""
    if not isinstance(thelist,list):
        return False
    
    # All float
    okay = True
    for x in thelist:
        if (not type(x) in [int,float]):
            okay = False
    
    return okay


def is_2D_list(thelist,dim):
    """Return: True if thelist is a 2D list of numbers (int or float)
    AND if the number of columns in thelist is equal to dim"""
    if not isinstance(thelist,list):
        return False
    
    result = True
    for row in thelist:
        if not is_point(row):
            result = False
        if len(row) != dim:
            result = False
    return result


def is_indices_list(thelist,k,ds):
    """Return: True if thelist is a list of k valid indices into ds"""
    
    result = True
    if len(thelist) != k:
        return False
    if not isinstance(thelist,list):
        return False
    for x in thelist:
        if x < 0 or x > ds.getSize():
            result = False
    return result
    

# CLASSES
class Dataset(object):
    """Instance is a dataset for k-means clustering.

    The data is stored as a list of list of numbers
    (ints or floats).  Each component list is a data point.

    Instance Attributes:
        _dimension [int > 0. Value never changes after initialization]:
            the point dimension for this dataset
        _contents  [a 2D list of numbers (float or int), possibly empty]:
            the dataset contents
        
    Additional Invariants:
        The number of columns in _contents is equal to _dimension.  That is,
        for every item _contents[i] in the list _contents, 
        len(_contents[i]) == dimension.
    
    None of the attributes should be accessed directly outside of the class
    Dataset (e.g. in the methods of class Cluster or KMeans). Instead, this class 
    has getter and setter style methods (with the appropriate preconditions) for 
    modifying these values.
    """
    
    
    def __init__(self, dim, contents=None):
        """Initializer: Makes a database for the given point dimension.
    
        The parameter dim is the initial value for attribute _dimension.  
  
        The optional parameter contents is the initial value of the
        attribute _contents. When assigning contents to the attribute
        _contents it COPIES the list contents. If contents is None, the 
        initializer assigns _contents an empty list. The parameter contents 
        is None by default.
    
        Precondition: dim is an int > 0. contents is either None or 
        it is a 2D list of numbers (int or float). If contents is not None, 
        then contents is not empty and the number of columns of contents is 
        equal to dim.
        """
        assert isinstance(dim, int), `dim` + ' is not an integer'
        assert dim > 0, `dim` + ' is not > 0'
        assert (contents == None or is_2D_list(contents,dim)), (`contents` +
            ' is neither None or a 2D list of numbers (int or float)'+
            ' and/or the number of columns of contents is not equal to'+`dim`)
        if contents != None:
            assert contents != [], `contents` + ' is an empty list'
        
        self._dimension = dim
        
        if contents == None:
            self._contents = []
        else:
            contentscopy = []
            numrows = len(contents)
            numcols = dim
            for row in range(numrows):
                rowcopy = []
                for col in range(numcols):
                    rowcopy.append(contents[row][col])
                contentscopy.append(rowcopy)
            self._contents = contentscopy
        #print '_contents is ' + `self._contents`
    
    
    def getDimension(self):
        """Return: The point dimension of this data set.
        """
        return self._dimension
    
    
    def getSize(self):
        """Return: the number of elements in this data set.
        """
        result = len(self._contents)
        return result
    
    
    def getContents(self):
        """Return: The contents of this data set as a list.
        
        This method returns the attribute _contents directly.  Any changes
        made to this list will modify the data set.  If you want to access
        the data set, but want to protect yourself from modifying the data,
        use getPoint() instead.
        """
        return self._contents
    
    
    def getPoint(self, i):
        """Returns: A COPY of the point at index i in this data set.
        
        Often, we want to access a point in the data set, but we want a copy
        to make sure that we do not accidentally modify the data set.  That
        is the purpose of this method.  
        
        If you actually want to modify the data set, use the method getContents().
        That returns the list storing the data set, and any changes to that
        list will alter the data set.
        While it is possible, to access the points of the data set via
        the method getContents(), that method 
        
        Precondition: i is an int that refers to a valid position in the data
        set (e.g. i is between 0 and getSize()).
        """
        assert isinstance(i, int), `i` + ' is not an integer'
        assert (i >= 0 and i < self.getSize()), `i`+' is not a valid position in the data'
        
        point = self._contents[i]
        
        pointcopy = []
        for x in point:
            pointcopy.append(x)
        return pointcopy
    
    
    def addPoint(self,point):
        """Adds a COPY of point at the end of _contents.
        
        This method does not add the point directly. It adds a copy of the point.
    
        Precondition: point is a list of numbers (int or float),  
        len(point) = _dimension.
        """
        assert is_point(point), `point`+' is not a list of numbers (int or float)'
        assert len(point) == self._dimension, ('length of ' + `point` +
                                          ' is not ' + `self._dimension`)
        
        pointcopy = []
        for x in point:
            pointcopy.append(x)
        
        self._contents.append(pointcopy)
    
    
    # PROVIDED METHODS: Do not modify!
    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._contents) 
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)


class Cluster(object):
    """An instance is a cluster, a subset of the points in a dataset.

    A cluster is represented as a list of integers that give the indices
    in the dataset of the points contained in the cluster.  For instance,
    a cluster consisting of the points with indices 0, 4, and 5 in the
    dataset's data array would be represented by the index list [0,4,5].

    A cluster instance also contains a centroid that is used as part of
    the k-means algorithm.  This centroid is an n-D point (where n is
    the dimension of the dataset), represented as a list of n numbers,
    not as an index into the dataset.  (This is because the centroid
    is generally not a point in the dataset, but rather is usually in between
    the data points.)

    Instance attributes:
        _dataset [Dataset]: the dataset this cluster is a subset of
        _indices [list of int]: the indices of this cluster's points in the dataset
        _centroid [list of numbers]: the centroid of this cluster
    Extra Invariants:
        len(_centroid) == _dataset.getDimension()
        0 <= _indices[i] < _dataset.getSize(), for all 0 <= i < len(_indices)
    """
    
    # Part A
    def __init__(self, ds, centroid):
        """A new empty cluster whose centroid is a copy of <centroid> for the
        given dataset ds.
    
        Pre: ds is an instance of a subclass of Dataset.
             centroid is a list of ds.getDimension() numbers.
        """
        assert isinstance(ds,Dataset)
        assert is_point(centroid), `centroid`+' is not a list of numbers'
        assert len(centroid) == ds.getDimension(), ('length of ' + `centroid` +
                                            ' is not ' + `ds.getDimension()`)
        
        self._dataset = ds
        self._indices = []
        
        centroidcopy = []
        numrows = len(centroid)
        for x in range(numrows):
            centroidcopy.append(centroid[x])
        self._centroid = centroidcopy


    def getCentroid(self):
        """Returns: the centroid of this cluster.
        
        This getter method is to protect access to the centroid.
        """
        return self._centroid
    
    
    def getIndices(self):
        """Returns: the indices of points in this cluster
        
        This method returns the attribute _indices directly.  Any changes
        made to this list will modify the cluster.
        """
        return self._indices
    
    
    def addIndex(self, index):
        """Add the given dataset index to this cluster.
        
        If the index is already in this cluster, this method leaves the
        cluster unchanged.
        
        Precondition: index is a valid index into this cluster's dataset.
        That is, index is an int in the range 0.._dataset.getSize().
        """
        
        assert isinstance(index, int), `index` + ' is not an integer'
        assert (index >= 0 and index <= self._dataset.getSize()), (`index`+
                                        ' is not a valid index in the dataset')
        
        if not index in self._indices:
            self._indices.append(index)
    
    
    def clear(self):
        """Remove all points from this cluster, but leave the centroid unchanged.
        """
        del self._indices[:]
    
    
    def getContents(self):
        """Return: a new list containing copies of the points in this cluster.
        
        The result is a list of list of numbers.  It has to be computed from
        the indices.
        """
        contentscopy = []
        for i in self._indices:
            point = self._dataset.getPoint(i)
            length = len(point)
            pointcopy = []
            for x in range(length):
                pointcopy.append(point[x])
            contentscopy.append(pointcopy)
        return contentscopy
    
    
    # Part B
    def distance(self, point):
        """Return: The euclidean distance from point to this cluster's centroid.
    
        Pre: point is a list of numbers (int or float)
             len(point) = _ds.getDimension()
        """
        assert is_point(point), `point` + 'is not a point'
        assert len(point) == self._dataset.getDimension(), (`point` +
                                            'does not have correct dimensions')

        dsqrd = 0
        centroid = self.getCentroid()
        length = self._dataset.getDimension()
        for pos in range(length):
            term = (point[pos] - centroid[pos])**2
            dsqrd = dsqrd + term
        dist = math.sqrt(dsqrd)
        return dist

    
    def updateCentroid(self):
        """Returns: Trues if the centroid remains the same after recomputation,
        and False otherwise.
        
        This method recomputes the _centroid attribute of this cluster. The
        new _centroid attribute is the average of the points of _contents
        (To average a point, average each coordinate separately).  
    
        Whether the centroid "remained the same" after recomputation is
        determined by numpy.allclose.  The return value should be interpreted
        as an indication of whether the starting centroid was a "stable"
        position or not.
    
        If there are no points in the cluster, the centroid. does not change.
        """
        cntnts = self.getContents()
        numpts = len(cntnts)
        dim = len(cntnts[0])
        
        total = []
        
        for pos in range(dim):
            termtot = 0.0
            for pindex in range(numpts):
                term = cntnts[pindex][pos]
                termtot = termtot + float(term)
            total.append(termtot)
        
        #print 'contents is'+`cntnts`
        #print 'total is '+`total`
        
        avg = []
        for pos in range(dim):
            avgterm = total[pos]/float(numpts)
            avg.append(avgterm)
        
        oldcentroid = self._centroid    
        self._centroid = avg
        
        if numpy.allclose(oldcentroid,avg):
            return True
        else:
            return False
    
    
    # PROVIDED METHODS: Do not modify!
    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._centroid)
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)


class ClusterGroup(object):
    """An instance is a set of clusters of the points in a dataset.

    Instance attributes:
        _dataset [Dataset]: the dataset which this is a clustering of
        _clusters [list of Cluster]: the clusters in this clustering (not empty)
    """
    
    # Part A
    def __init__(self, ds, k, seed_inds=None):
        """A clustering of the dataset ds into k clusters.
        
        The clusters are initialized by randomly selecting k different points
        from the database to be the centroids of the clusters.  If seed_inds
        is supplied, it is a list of indices into the dataset that specifies
        which points should be the initial cluster centroids.
        
        Pre: ds is an instance of a subclass of Dataset.
             k is an int, 0 < k <= ds.getSize().
             seed_inds is None, or a list of k valid indices into ds.
        """
        assert isinstance(ds,Dataset), `ds`+' is not a subclass of Dataset'
        assert isinstance(k,int), `k`+ ' is not an integer'
        assert (k > 0 and k <= ds.getSize()), `k`+' is not in allowed range'
        assert (seed_inds == None or is_indices_list(seed_inds,k,ds)), (`seed_inds`+
                            +'is not None or a list of k valid indices into ds')
        
        self._dataset = ds
        
        if seed_inds == None:
            indices = random.sample(range(ds.getSize()),k)
        else:
            indices = seed_inds
            
        centroidlist = []
        dim = ds.getDimension()
        for pindex in indices:
            point = ds.getPoint(pindex)
            centroidlist.append(point)
            
        clusterslist = []
        for centroid in centroidlist:
            cluster = Cluster(ds, centroid)
            clusterslist.append(cluster)
        self._clusters = clusterslist
        
    
    def getClusters(self):
        """Return: The list of clusters in this object.
        
        This method returns the attribute _clusters directly.  Any changes
        made to this list will modify the set of clusters.
        """ 
        return self._clusters


    # Part B
    def _nearest_cluster(self, point):
        """Returns: Cluster nearest to point
    
        This method uses the distance method of each Cluster to compute
        the distance between point and the cluster centroid. It returns
        the Cluster that is the closest.
        
        Ties are broken in favor of clusters occurring earlier in the
        list of self._clusters.
        
        Pre: point is a list of numbers (int or float),
             len(point) = self._dataset.getDimension().
        """
        
        assert is_point(point), `point` + 'is not a point'
        assert len(point) == self._dataset.getDimension(), (`point`+'is not'+
                                                        +'the correct length')
        
        #arbitrarily sets first cluster in set as the nearest cluster
        nearestcluster = self.getClusters()[0]
        distance = nearestcluster.distance(point)
        
        for cluster in self.getClusters():
            newdist = cluster.distance(point)
            if newdist < distance:
                distance = newdist
                nearestcluster = cluster
                
        return nearestcluster

    
    def _partition(self):
        """Repartition the dataset so each point is in exactly one Cluster.
        """
        # First, clear each cluster of its points.  Then, for each point in the
        # dataset, find the nearest cluster and add the point to that cluster.
        
        #print 'cluster set is ' + `self.getClusters()`
        
        for cluster in self.getClusters():
            #print 'cluster centroid is '+`cluster.getCentroid()`
            #print 'cluster indices is '+`cluster.getIndices()`
            cluster.clear()
            #print 'after clearing cluster indices is '+`cluster.getIndices()`
        
        #print 'after clearing, cluster set is ' + `self.getClusters()`
        
        dim = self._dataset.getSize()
        #print 'dim is '+`dim`
        for index in range(dim):
            point = self._dataset.getPoint(index)
            #print 'point is'+`point`
            nearestcluster = self._nearest_cluster(point)
            #print 'nearest cluster is '+`nearestcluster`
            nearestcluster.addIndex(index)
            #print 'new nearestcluster indices is '+`nearestcluster.getIndices()`

    
    # Part C
    def _update(self):
        """Return:True if all centroids are unchanged after an update; False otherwise.
        
        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It then returns the appropriate value.
        """
        
        unchanged = True
        
        for cluster in self.getClusters():
            #print 'before change, cluster centroid is '+`cluster.getCentroid()`
            unchanged = cluster.updateCentroid()
            #print 'after change, cluster centroid is '+`cluster.getCentroid()`
        
        return unchanged

    
    def step(self):
        """Return: True if the algorithm converges after one step; False otherwise.
        
        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns the appropriate value.
        """
        # In a cycle, we partition the points and then update the means.
        
        self._partition()
        return self._update()

    
    # Part D
    def run(self, maxstep):
        """Continue clustering until either it converges or maxstep steps 
        (which ever comes first).
        Precondition: maxstep is an int >= 0.
        """
        # Call step repeatedly, up to maxstep times, until the algorithm
        # converges.  Stop after maxstep iterations even if the algorithm has not
        # converged.
        
        # IMPLEMENT ME
        assert isinstance(maxstep, int), `maxstep` +' is not an integer'
        assert maxstep >= 0, `maxstep` + ' is not >= 0'
        
        count = 0
        convergence = False
        
        while count < maxstep and convergence == False:
            convergence = self.step()
            count = count + 1

    
    # PROVIDED METHODS: Do not modify!
    def __str__(self):
        """Returns: String representation of the centroid of this cluster."""
        return str(self._clusters)
    
    def __repr__(self):
        """Returns: Unambiguous representation of this cluster. """
        return str(self.__class__) + str(self)

