import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataSet(object):
    
    """Object class that holds a data set, which can be imported from a data file."""
    
    def __init__(self):
        """Instantiate an instance of the DataSet class"""
        
        self.filename = None
        self.data = None
        self.xdata = None
        self.ydata = None
        
    def load_data(self, filename, ftype):
        """Read the data csv file and build a DataSet object to hold the data"""
        
        self.filename = filename
        
        if ftype == "csv":
            self.data = pd.read_csv(self.filename, sep=",", header=None)
        elif ftype == "tsv":
            self.data = pd.read_csv(self.filename, sep="\t", header=None)
        else:
            raise Exception("File type is not supported.")
            
        self.xdata = np.array(self.data[0])
        self.ydata = np.array(self.data[1])
        
    def assign_data(self, xdata, ydata):  
        "Allows data to be assigned to the dataset object from an array, etc. Doesn't require a file."
        
        self.xdata = xdata
        self.ydata = ydata
    
    def plot_data(self):
        """Function to generate a quick x vs. y plot of the data held in the DataSet object"""
        
        plt.plot(self.xdata, self.ydata)
        plt.title("Plot of the data")
        
    
