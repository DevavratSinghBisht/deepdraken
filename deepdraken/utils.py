import numpy as np

def one_hot(class_idx: int, total_classes: int) -> np.ndarray:
    '''
        Performs one hot enoding on the index.
        
        :param class_idx: index of the selected class
        :param total_classes: total number of classes
        :returns: 2D aray of one hot encoded indices.
    '''
    
    index = np.asarray(class_idx)
    
    if len(index.shape) == 0:
        index = np.asarray([index])
    
    assert len(index.shape) == 1
    
    num = index.shape[0]
    output = np.zeros((num, total_classes), dtype=np.float32)
    output[np.arange(num), index] = 1
    
    return output

def one_hot_if_needed(class_idx, total_classes) -> np.ndarray:
    '''
        Checks if the labels are already one hot encoded.
        Performs One hot encoding only if the labels are not already onehot encoded.

        :param class_idx: index of the selected class
        :param total_classes: total number of classes
        :returns: 2D aray of one hot encoded indices.
    '''
    label = np.asarray(class_idx)
    
    if len(label.shape) <= 1:
        label = one_hot(class_idx, total_classes)
    
    assert len(label.shape) == 2
    return label
