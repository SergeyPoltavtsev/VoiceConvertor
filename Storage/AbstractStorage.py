class AbstractStorage(object):
    """
    Abstract storage which lists the methods that are expected to be supplied.
    The class is used for low level manipulations with a dataset as reading and writing to the hard disk.
    However, it can be more generic.
    """

    def __init__(self, path):
        """
        Initializes the storage

        Inputs:
        - path: a path to a storage file (should have .tfrecords extention)
        """
        self.path = path

    def InsertRow(self, row):
        """
        Method which saves an item into the storage
        Note: expected that an item should be stored as one row
        """
        raise NotImplementedError("Should have implemented this")

    def Inputs(self):
        """
        Method return the next mini batch from the dataset.
        """
        raise NotImplementedError("Should have implemented this")
