import numpy as np

def one_hot(self, class, total_classes):
    '''
        Performs one hot enoding on the index.
        
        Params:
            index: index of the object to be generated.
        Returns: 
            output: 2D aray of one hot encoded indices.
    '''
    
    index = np.asarray(class)
    
    if len(index.shape) == 0:
        index = np.asarray([index])
    
    assert len(index.shape) == 1
    
    num = index.shape[0]
    output = np.zeros((num, total_classes), dtype=np.float32)
    output[np.arange(num), index] = 1
    
    return output

def one_hot_if_needed(self, class, total_classes):
    '''
        Checks if the labels are already one hot encoded.
        Performs One hot encoding only if the labels are not already onehot encoded.
    '''
    label = np.asarray(class)
    
    if len(label.shape) <= 1:
        label = self.one_hot(class, total_classes)
    
    assert len(label.shape) == 2
    return label
