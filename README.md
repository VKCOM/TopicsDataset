## TopicsDataset

The TopicsDataset is a dataset comprised of [VK.com](http://vk.com/) social media posts sorted into different topics.
Each data point consists of two post features: embedding (image and/or text) and topic ID (numbered from 0 to 49). All data is anonymous.

The Experiments folder of the repository holds the source code of our Active Learning research experiments conducted on this data. Our approach is based on [https://github.com/modAL-python/modAL](https://github.com/modAL-python/modAL).

Experiments/torch_topics_uncertainty.py provides a basic example of comparing different active learning strategies applied to the TopicsDataset.

## Download dataset

Download link (1.1 GB): [https://vk.cc/c5EptY](https://vk.cc/c5EptY)
The dataset is in .npy format, consisting of a two-dimensional array of floats. Refer to experiments/datasets/topics_ds.py for an example of how to read this data format.