# Storing CT slices and corresponding ground truth labels in h5 file

import h5py


class HDF5DatasetWriter:

    def __init__(self, img_shape, label_shape, output_path, buf_size=200):
        '''
            buf_size: flush the data to external memory if there is buf_size data stored in memory
        '''
        self.db = h5py.File(output_path, 'w')
        self.images = self.db.create_dataset('images', img_shape, maxshape=(None,) + img_shape[1:], dtype='float')
        self.labels = self.db.create_dataset('labels', label_shape, maxshape=(None,) + label_shape[1:], dtype='int')

        self.buf_size = buf_size
        self.buffer = {'images': [], 'labels': []}
        self.idx = 0

    def add(self, images, labels):

        self.buffer['images'].extend(images)
        self.buffer['labels'].extend(labels)
        print('Length of added data:', len(self.buffer['images']))

        if len(self.buffer['images']) >= self.buf_size:
            self.flush()

    def flush(self):

        i = self.idx + len(self.buffer['images'])
        if i > self.images.shape[0]:
            self.images.resize(i, axis=0)
            self.labels.resize(i, axis=0)

        self.images[self.idx:i] = self.buffer['images']
        self.labels[self.idx:i] = self.buffer['labels']
        print('h5py has written %d data' % i)

        self.idx = i
        self.buffer = {'images': [], 'labels': []}

    def close(self):

        if len(self.buffer['images']) > 0:
            self.flush()

        self.db.close()
        return self.idx