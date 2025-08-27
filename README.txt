Pytorch notes

This example will use the FashionMNIST dataset in the TorchVision library. It contains 70k grayscale images of fashion items sizing ~40mb. 

Calling down the data involves utilizing the Dataset constructor which takes the arguments:
	- root: takes a string directory giving the target storage for the dataset. If the string does not find a 		matching directory, one will be created.
	- training: takes Boolean and determines whether do get training labels with the data (T) or not (F)
	- download: takes Boolean and determines if data should be downloaded if not found (T) or not (F)
	- transform: takes a callable function that must return an appropriate object (an image in this case)