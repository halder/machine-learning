# information entropy: "average rate at which information is produced by a stochastic source of data"
def entropy(y):
    N = len(y)
    s1 = (y == 1).sum() # binary classification tree; in this ex. "y=1" is to be classified
    # events that always occur do not communicate information, hence if all values of y are the same, the entropy is 0
    if s1 == 0 or s1 == N: # shortcut without any calculation
        return 0
    p = s1 / N
    q = 1 - p
    return -p*np.log2(p)-q*np.log2(q)
    
# Single Node; Wrapped in a DecisionTree wrapper
class TreeNode():
    # each Node must know it's own depth relative to the max_depth to avoid endless recursion
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        
    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1: # base cases: if there is only 1 sample or multiple sample but only 1 target value
            self.col = None
            self.split = None
            self.left = None # indicates no split
            self.right = None # indicates no split
            self.prediction = Y[0] # if base case == True: predict the 1 label
        else:
            D = X.shape[1]
            columns = range(D) # N, D = X.shape[0], X.shape[1] 
            
            # find best (initial; on first recursion) split
            max_ig = 0
            best_split = None
            best_col = None
            
            for col in columns: # iterate over all columns -> which column is best suited for the split
                # find_split should find the best split for each col internally; compare splitting at different boundaries
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig: # set values for the first split according to highest information gain 
                    max_ig = ig
                    best_split = split
                    best_col = col                
                if max_ig == 0: # if IG == 0 nothing is gained from splitting; base case
                    self.col = None
                    self.split = None
                    self.left = None # indicates no split
                    self.right = None # indicates no split
                    self.prediction = np.round(Y.mean()) # np.argmax(Y) # np.round(Y.mean()) ?
                else:
                    self.col = best_col
                    self.split = best_split                 
                    # last base case: if we are at max_depth, we will not split anymore
                    if self.depth == self.max_depth:
                        #print("MAX DEPTH REACHED")
                        self.left = None # indicates no split
                        self.right = None # indicates no split
                        # two predictions, one for left_side, one for right_side
                        # take majority class after splitting the data
                        self.prediction = [
                            # Y[X[:, best_col]] - Y's for all rows in X, with column = best_col
                            # conditioning on self.split is essentially Y[:self.split] ?
                            np.round(Y[X[:, best_col] < self.split].mean()), # < indicates LEFT --- why mean() ?
                            np.round(Y[X[:, best_col] >= self.split].mean()) # >= RIGHT
                        ]
                    else:
                        # NOT a base case, do a recursion -> We will now go one step deeper into the Tree
                        # left index == 0:best_split; right == best_split:end;
                        '''
                        Why left/right instead of [:best_split]/[best_split:]
                        
                        Each column is being looked at separately, visualize the process of splitting at boundaries as
                        the following:
                        
                                                    ---------------------------------
                            (random values)   X:    | 2 ' 5 ' 2 ' 5 ' 5 ' 2 ' 2 ' 3 |
                                              Y:    | 1 ' 1 ' 1 ' 0 ' 0 ' 1 ' 1 ' 1 | 
                                                    ---------------------------------
                                    
                        ``best_split`` is calculated by splitting at the boundaries between 1's and 0's in Y and calculating
                        the respective information gain.
                        
                        Assume a best_split has been found, the best_col (X) now needs to be split into 2 to create two new
                        TreeNodes and start another recursion.
                        If visualized as above, the column X will no be split into Xleft and Xright at the best_split position.
                        '''
                        left_index = (X[:, best_col] < best_split)
                        right_index = (X[:, best_col] >= best_split)
                        
                        # set X/Y for left and right child nodes
                        Xleft = X[left_index]
                        Yleft = Y[left_index]
                        
                        Xright = X[right_index]
                        Yright = Y[right_index]
                        
                        # create left and right child nodes, we are now one step deeper. 
                        self.left = TreeNode(self.depth+1, self.max_depth)
                        self.right = TreeNode(self.depth+1, self.max_depth)
                        # call fit recursively on the child nodes. From this point on, more and more child nodes will be
                        # created recursively depending on information_gain and the depth relative to the max_depth
                        self.left.fit(Xleft, Yleft)
                        self.right.fit(Xright, Yright)   
                        
    def find_split(self, X, Y, col):
        '''
        get boundaries of 0/1
        calculate information gain when splitting at each boundary
        return information gain and split_index
        '''
        x_values = X[:, col] # dummy used to sort the current col
        sorted_index = np.argsort(x_values) # argsort returns indices that would sort the array
        # set x, y to sorted versions of themselves
        x_values, y_values = x_values[sorted_index], Y[sorted_index]
        
        '''
        Don't know why use the conditioning; why [0]
        '''
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        
        # iterate over all found boundaries, calculate information gain one by one
        for i in boundaries:
            split = (x_values[i] + x_values[i+1]) / 2 # works since x_values is sorted 
            ig = self.information_gain(x_values, y_values, split)            
            if ig > max_ig:
                max_ig = ig
                best_split = split                
        return max_ig, best_split        
    
    def information_gain(self, X, Y, split):
        # split Y into left and right part; condition on sorted Y, X
        y0 = Y[X < split] 
        y1 = Y[X >= split]
        N = len(Y)        
        if len(y0) == 0 or len(y0) == N: # same as for entropy
            return 0        
        p0 = len(y0) / N
        p1 = 1 - p0        
        return entropy(Y) - p0*entropy(y0) - p1*entropy(y1)
        
    def predict_one(self, X): # predicts one column
        if self.col is not None and self.split is not None: # is 1 condition enough? Both mean that there was a split
            feature = X[self.col]
            '''
            Don't know yet why feature [type: array] < split [type: float]
            '''
            if feature < self.split: # smaller self.split == "left"
                if self.left: # if this is a left ChildNode
                    p = self.left.predict_one(X) # recursively call predict_one to go deeper
                else: # if this is a LeafNode
                    p = self.prediction[0] # [0] was declared as "left" in fit()
            else: # >= self.split == "right"
                if self.right:
                    p = self.right.predict_one(X)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p
    
    def predict(self, X): # calls predict_one for each individual x (=each column)
        N = len(X)
        P = np.zeros(N)        
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P    
        
# Wrapper class
class DecisionTree:
    def __init__(self, max_depth=None): # DT only has max_depth; depth are TreeNode attributes
        self.max_depth = max_depth
    
    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth) # fit RootNode
        self.root.fit(X, Y)
        
    def predict(self, X):
        return self.root.predict(X)
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y), P        
