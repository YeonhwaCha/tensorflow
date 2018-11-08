# Load Dataset
import glob
import lmdb
import os
import re
import tensorflow as tf

from dataset.dataset_yolo2 import augment_tfrecord, decode_lmdb, decode_tfrecord


##########################################################
# Local Methods
# pattern : file_pattern-[num].tfrecord 
def _get_num_samples(tfrecord_file):
    pattern = re.search("-\d+.", tfrecord_file)
    num     = re.search("-\d+.", tfrecord_file).group(0)
    num     = int(num[1:-1])
    return num


##########################################################
# BASE Class is for read_and_decode
class ReadAndDecodeModel(object):
    def __init__(self, dataset_dir, dataset_type, num_batches, num_epochs, shuffle):
        # check the mendatory parameters
        if (dataset_dir is not None):
            self.read_and_decode(dataset_dir, dataset_type, num_batches, num_epochs, shuffle)
    
    def get_iterator(self):
        return self.iterator

    def read_and_decode(self):
        raise NotImplementedError("read_and_decode need to be defined...")



###########################################################
# dataset type : lmdb
# python r1.11 : LMDBDataset
class LmdbReadAndDecode(ReadAndDecodeModel):
    def __init__(self, dataset_dir, dataset_type, num_batches=16, num_epochs=1000, shuffle=False):
        super(LmdbReadAndDecode, self).__init__(dataset_dir, dataset_type, num_batches, num_epochs, shuffle)


    def read_and_decode(self, dataset_dir, dataset_type, num_batches, num_epochs, shuffle):
        dataset = tf.contrib.data.LMDBDataset(dataset_dir)
        dataset = dataset.map(decode_lmdb)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(num_batches, drop_remainder=True)
        
        self.iterator = dataset.make_one_shot_iterator()
   

###########################################################
# dataset type : TFRecord
# Python over r1.10 : TFRecordDataset 
class TFRecordReadAndDecode(ReadAndDecodeModel):
    def __init__(self, dataset_dir, dataset_type='train', num_batches=16, num_epochs=1000, shuffle=False):
        super(TFRecordReadAndDecode, self).__init__(dataset_dir, dataset_type, num_batches, num_epochs, shuffle)
   

    def read_and_decode(self, dataset_dir, dataset_type, file_pattern, num_batches, num_epochs, shuffle):
        _source_files = glob.glob(os.path.join(dataset_dir, file_pattern % dataset_type))
        _num_samples  = sum([_get_num_samples(tf_file) for tf_file in _source_files])
        
        dataset = tf.data.TFRecordDataset(_source_files)
        dataset = dataset.map(decode_tfrecord)
        dataset = dataset.map(augment_tfrecord)
        if shuffle == True:
            dataset = dataset.shuffle(_num_samples)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(num_batches, drop_remainder=True)
        
        self.iterator = dataset.make_one_shot_iterator()

