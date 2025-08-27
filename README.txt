Pytorch notes

This example will use the FashionMNIST dataset in the TorchVision library. It contains 70k grayscale images of fashion items sizing ~40mb. 

Calling down the data involves utilizing the Dataset constructor which takes the arguments:
	- root: takes a string directory giving the target storage for the dataset. If the string does not find a
		matching directory, one will be created.
	- training: takes Boolean and determines whether do get training labels with the data (T) or not (F)
	- download: takes Boolean and determines if data should be downloaded if not found (T) or not (F)
	- transform: takes a callable function that must return an appropriate object (an image in this case)

Each dataset gets plugged into a data loader to parse it and make it iterable. The dataloader constructor has many options:
	- Dataset: the only mandatory parameter, this takes a torch dataset object
	- batch_size: takes an int, determines the number of samples to be loaded per batch
	- shuffle: takes a bool, option to shuffle data after every batch/epoch
	- sampler: takes an iterable, mutex with shuffle, defines the sample draw strategy from the dataset
	- batch_sampler: takes an iterable, mutex with batch_size, shuffle, sampler, and drop_last, works like sampler
		but returns indices in batches
	- num_workers: takes an int, specifies the number of subprocesses for loading
	- collate_fn: takes a callable, given a list of samples, returns a list of tensors from the callable, used
		with map-style data
	- pin_memory: takes a bool, specifies to use non-pageable CPU memory for copies of tensors
	- drop_last: takes a bool, truncates the remainder of the final batch of data if the dataset size/number of
		batches has a non-zero remainder
	- timeout: takes a numeric expression, specifies a timeout value for batch collection
	- worker_init_fn: takes a callable, applies the callable to every worker after seeding but before data load
	- multiprocessing_context: takes a str, specifies multiprocessor behavior (default is #noqa D401)
	- generator: takes a torch.Generator object, specifies the RNG for indices and worker base_seeds
	- prefetch_factor: takes an int, number of advance-loaded batches per worker. IF num_workers > 0, default is
		2, otherwise default value is 0
	- persistent_workers: takes a bool, tells whether to keep worker process alive after dataset consumption
	- pin_memory_device: takes a str, points to device to pin_memory, slated for deprecation
	- in_order: takes a bool, usually batches return in first-in-first-out, set this to false to return whenever 
		they're done

Important to note that the dataloader doesn't do anything other than set up the dataset for workover by pytorch. It structures it and allows for some metadata to be exposed (such as .shape that shows the dimensions of the tensors of the dataset)

A tensor is the basic data unit of pytorch. It has n-dimnensionality dependent on the type of data being looked at. For example, the torchvision>FashionMNIST dataset has four dimensional tensors. The first dimension is always the batch size. the next three for the visual tensors are <number of color channels> i.e. 1 for grayscale and 3 for RGB, height in pixels of an image, and width for the same.
