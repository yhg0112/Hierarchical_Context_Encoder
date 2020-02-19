from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.text_problems import VocabType
from tensor2tensor.utils import registry as t2t_registry

from .corpus_dataset import ParallelCorpusProblem, ParallelCorpusProblemWithContexts

SOURCE_TRAIN_FILES = ['raw_corpus/IWSLT17/de-en/train.tags.de-en.de']
TARGET_TRAIN_FILES = ['raw_corpus/IWSLT17/de-en/train.tags.de-en.en']

SOURCE_DEV_FILES = ['raw_corpus/IWSLT17/de-en/IWSLT17.TED.dev2010.de-en.de.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2010.de-en.de.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2011.de-en.de.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2012.de-en.de.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2013.de-en.de.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2014.de-en.de.xml']
TARGET_DEV_FILES = ['raw_corpus/IWSLT17/de-en/IWSLT17.TED.dev2010.de-en.en.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2010.de-en.en.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2011.de-en.en.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2012.de-en.en.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2013.de-en.en.xml',
                    'raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2014.de-en.en.xml']

SOURCE_TEST_FILES = ['raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2015.de-en.de.xml']
TARGET_TEST_FILES = ['raw_corpus/IWSLT17/de-en/IWSLT17.TED.tst2015.de-en.en.xml']


@t2t_registry.register_problem
class DeEnIwslt17(ParallelCorpusProblem):
    @property
    def approx_vocab_size(self):
        return 2**14

    @property
    def source_compiled_corpus_filename(self):
        return ["train_source.de", "dev_source.de", "test_source.de"]

    @property
    def target_compiled_corpus_filename(self):
        return ["train_target.en", "dev_target.en", "test_target.en"]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TRAIN_FILES,
                                                         self.source_compiled_corpus_filename[0])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TRAIN_FILES,
                                                         self.target_compiled_corpus_filename[0])
        elif dataset_split == problem.DatasetSplit.EVAL:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_DEV_FILES,
                                                         self.source_compiled_corpus_filename[1])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_DEV_FILES,
                                                         self.target_compiled_corpus_filename[1])
        elif dataset_split == problem.DatasetSplit.TEST:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TEST_FILES,
                                                         self.source_compiled_corpus_filename[2])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TEST_FILES,
                                                         self.target_compiled_corpus_filename[2])

        return text_problems.text2text_txt_iterator(os.path.join(data_dir, source_file_name),
                                                    os.path.join(data_dir, target_file_name))

    def compile_corpus_files(self, data_dir, file_list, out_filename):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        root_dir = os.path.abspath(os.path.join(data_dir, os.pardir))

        out_file_path = os.path.join(data_dir, out_filename)

        if tf.gfile.Exists(out_file_path):
            return out_filename

        with tf.gfile.Open(out_file_path, 'w') as compiled_corpus:
            for file_path in file_list:
                tf.logging.info("Checking out %s" % file_path)
                with tf.gfile.Open(os.path.join(root_dir, file_path), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line[0] is not '<':
                            compiled_corpus.write(line.strip())
                            compiled_corpus.write('\n')
                            continue
                        tag = line.split('>')[0]
                        line = line.split('>')[1].split('<')[0].strip()
                        if 'description' in tag or 'keywords' in tag or 'title' in tag or 'seg id' in tag:
                            if line is not '':
                                compiled_corpus.write(line.strip())
                                compiled_corpus.write('\n')
                                continue

        return out_filename


@t2t_registry.register_problem
class EnDeIwslt17(DeEnIwslt17):

    @property
    def vocab_filename(self):
        if self.vocab_type == VocabType.SUBWORD:
            return "vocab.%s.%d.%s" % (super(EnDeIwslt17, self).name,
                                       self.approx_vocab_size,
                                       VocabType.SUBWORD)
        else:
            return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

    @property
    def source_compiled_corpus_filename(self):
        return ["train_source.en", "dev_source.en", "test_source.en"]

    @property
    def target_compiled_corpus_filename(self):
        return ["train_target.de", "dev_target.de", "test_target.de"]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TRAIN_FILES,
                                                         self.source_compiled_corpus_filename[0])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TRAIN_FILES,
                                                         self.target_compiled_corpus_filename[0])
        elif dataset_split == problem.DatasetSplit.EVAL:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_DEV_FILES,
                                                         self.source_compiled_corpus_filename[1])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_DEV_FILES,
                                                         self.target_compiled_corpus_filename[1])
        elif dataset_split == problem.DatasetSplit.TEST:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TEST_FILES,
                                                         self.source_compiled_corpus_filename[2])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TEST_FILES,
                                                         self.target_compiled_corpus_filename[2])

        return text_problems.text2text_txt_iterator(os.path.join(data_dir, source_file_name),
                                                    os.path.join(data_dir, target_file_name))


@t2t_registry.register_problem
class DeEnIwslt17WithContexts(ParallelCorpusProblemWithContexts):

    def __init__(self, was_reversed=False, was_copy=False):
        super(DeEnIwslt17WithContexts, self).__init__(was_reversed=was_reversed, was_copy=was_copy)
        self._num_sentences = 2

    @property
    def approx_vocab_size(self):
        return 2**14

    @property
    def source_compiled_corpus_filename(self):
        return ["train_source.de", "dev_source.de", "test_source.de"]

    @property
    def target_compiled_corpus_filename(self):
        return ["train_target.en", "dev_target.en", "test_target.en"]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TRAIN_FILES,
                                                         self.source_compiled_corpus_filename[0])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TRAIN_FILES,
                                                         self.target_compiled_corpus_filename[0])
        elif dataset_split == problem.DatasetSplit.EVAL:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_DEV_FILES,
                                                         self.source_compiled_corpus_filename[1])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_DEV_FILES,
                                                         self.target_compiled_corpus_filename[1])
        elif dataset_split == problem.DatasetSplit.TEST:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TEST_FILES,
                                                         self.source_compiled_corpus_filename[2])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TEST_FILES,
                                                         self.target_compiled_corpus_filename[2])

        return self.text_iterator_with_context(os.path.join(data_dir, source_file_name),
                                               os.path.join(data_dir, target_file_name))

    def compile_corpus_files(self, data_dir, file_list, out_filename):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        root_dir = os.path.abspath(os.path.join(data_dir, os.pardir))

        out_file_path = os.path.join(data_dir, out_filename)

        if tf.gfile.Exists(out_file_path):
            return out_filename

        with tf.gfile.Open(out_file_path, 'w') as compiled_corpus:
            for file_path in file_list:
                tf.logging.info("Checking out %s" % file_path)
                with tf.gfile.Open(os.path.join(root_dir, file_path), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line[0] is not '<':
                            compiled_corpus.write(line.strip())
                            compiled_corpus.write('\n')
                            continue
                        tag = line.split('>')[0]
                        line = line.split('>')[1].split('<')[0].strip()
                        if 'description' in tag or 'keywords' in tag or 'title' in tag or 'seg id' in tag:
                            if line is not '':
                                compiled_corpus.write(line.strip())
                                compiled_corpus.write('\n')
                                continue

        return out_filename


@t2t_registry.register_problem
class EnDeIwslt17WithContexts(DeEnIwslt17WithContexts):

    def __init__(self, was_reversed=False, was_copy=False):
        super(EnDeIwslt17WithContexts, self).__init__(was_reversed=was_reversed, was_copy=was_copy)
        self._num_sentences = 2

    @property
    def vocab_filename(self):
        if self.vocab_type == VocabType.SUBWORD:
            return "vocab.%s.%d.%s" % (super(EnDeIwslt17WithContexts, self).name,
                                       self.approx_vocab_size,
                                       VocabType.SUBWORD)
        else:
            return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

    @property
    def source_compiled_corpus_filename(self):
        return ["train_source.en", "dev_source.en", "test_source.en"]

    @property
    def target_compiled_corpus_filename(self):
        return ["train_target.de", "dev_target.de", "test_target.de"]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TRAIN_FILES,
                                                         self.source_compiled_corpus_filename[0])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TRAIN_FILES,
                                                         self.target_compiled_corpus_filename[0])
        elif dataset_split == problem.DatasetSplit.EVAL:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_DEV_FILES,
                                                         self.source_compiled_corpus_filename[1])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_DEV_FILES,
                                                         self.target_compiled_corpus_filename[1])
        elif dataset_split == problem.DatasetSplit.TEST:
            source_file_name = self.compile_corpus_files(data_dir,
                                                         TARGET_TEST_FILES,
                                                         self.source_compiled_corpus_filename[2])
            target_file_name = self.compile_corpus_files(data_dir,
                                                         SOURCE_TEST_FILES,
                                                         self.target_compiled_corpus_filename[2])

        return self.text_iterator_with_context(os.path.join(data_dir, source_file_name),
                                               os.path.join(data_dir, target_file_name))


@t2t_registry.register_problem
class EnDeIwslt17WithRandomizedContexts(EnDeIwslt17WithContexts):
    # this problem is for test only
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoders = self.get_or_create_vocab(data_dir, tmp_dir)
        return self.text2text_generate_encoded_with_randomized_contexts(sample_generator=generator,
                                                                        encoders=encoders,
                                                                        dataset_split=dataset_split)


@t2t_registry.register_problem
class DeEnIwslt17WithRandomizedContexts(DeEnIwslt17WithContexts):
    @property
    def vocab_filename(self):
        if self.vocab_type == VocabType.SUBWORD:
            return "vocab.%s.%d.%s" % (super(DeEnIwslt17WithRandomizedContexts, self).name,
                                       self.approx_vocab_size,
                                       VocabType.SUBWORD)
        else:
            return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

    # this problem is for test only
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoders = self.get_or_create_vocab(data_dir, tmp_dir)
        return self.text2text_generate_encoded_with_randomized_contexts(sample_generator=generator,
                                                                        encoders=encoders,
                                                                        dataset_split=dataset_split)
