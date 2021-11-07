# Data generator for training using model.fit_generator

import h5py
import numpy as np


class HDF5DatasetGenerator:

    def __init__(self, db_path, batch_size):

        self.batch_size = batch_size
        self.db = h5py.File(db_path)
        self.image_nums = self.db['images'].shape[0]
        self.batches_per_epoch = self.image_nums // batch_size

        print('Total images:', self.image_nums)


    def generator(self, epoches=np.inf):

        epoch = 0
        while epoch < epoches:
            # upset images
            shuffle_indices = np.arange(self.image_nums)
            shuffle_indices = np.random.permutation(shuffle_indices)
            for batch in range(self.batches_per_epoch):
                start_index = batch * self.batch_size
                end_index = min((batch+1) * self.batch_size, self.image_nums)

                # get serial number of one batch
                # sorted in increasing order
                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))

                images = self.db['images'][batch_indices]
                labels = self.db['labels'][batch_indices]

                yield (images, labels)

            epoch += 1


    def close(self):

        self.db.close()