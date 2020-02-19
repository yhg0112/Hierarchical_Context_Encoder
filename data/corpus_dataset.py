from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from random import randint

import tensorflow as tf

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry as t2t_registry


class ParallelCorpusProblem(translate.TranslateProblem):
    def is_generate_per_split(self):
        return True

    @property
    def approx_vocab_size(self):
        raise NotImplementedError

    @property
    def source_vocab_filename(self):
        return self.vocab_filename + '.source'

    @property
    def target_vocab_filename(self):
        return self.vocab_filename + '.target'

    @property
    def source_compiled_corpus_filename(self):
        raise NotImplementedError

    @property
    def target_compiled_corpus_filename(self):
        raise NotImplementedError

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1
        }]

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        data_dir = os.path.join(data_dir, self.name)

        filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
        }

        split_paths = []
        test_paths = []
        for split in self.dataset_splits:
            if split["split"] is not problem.DatasetSplit.TEST:
                split_paths.append((split["split"],
                                    filepath_fns[split["split"]](data_dir,
                                                                 split["shards"],
                                                                 shuffled=False)))
            else:
                test_paths.append((split["split"],
                                   filepath_fns[split["split"]](data_dir,
                                                                split["shards"],
                                                                shuffled=True)))

        all_paths = []
        for _, paths in split_paths:
            all_paths.extend(paths)

        if self.is_generate_per_split:
            for split, paths in split_paths:
                generator_utils.generate_files(
                    self._maybe_pack_examples(
                        self.generate_encoded_samples(data_dir, tmp_dir, split)), paths)
        else:
            generator_utils.generate_files(
                self._maybe_pack_examples(
                    self.generate_encoded_samples(
                        data_dir, tmp_dir, problem.DatasetSplit.TRAIN)), all_paths)

        generator_utils.shuffle_dataset(all_paths)

        test_split_paths = []
        for _, paths in test_paths:
            test_split_paths.extend(paths)

        if self.is_generate_per_split:
            for split, paths in test_paths:
                generator_utils.generate_files(
                    self._maybe_pack_examples(
                        self.generate_encoded_samples(data_dir, tmp_dir, split)), paths)
        else:
            generator_utils.generate_files(
                self._maybe_pack_examples(
                    self.generate_encoded_samples(
                        data_dir, tmp_dir, problem.DatasetSplit.TEST)), test_split_paths)

    @property
    def name(self):
        t2t_registry.default_name(self.__class__)

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
                is_sgm = file_path.endswith('.sgm')
                if is_sgm:
                    with tf.gfile.Open(os.path.join(root_dir, file_path), 'r') as f:
                        for line in f:
                            line = line.strip()
                            tag = line.split('>')[0]
                            if 'seg id' in tag:
                                line = line.split('>')[1].split('<')[0].strip()
                                compiled_corpus.write(line.strip())
                                compiled_corpus.write('\n')
                else:
                    with tf.gfile.Open(os.path.join(root_dir, file_path), 'r') as f:
                        for line in f:
                            compiled_corpus.write(line.strip())
                            compiled_corpus.write('\n')

        return out_filename

    def feature_encoders(self, data_dir):
        encoders = self.get_or_create_vocab(data_dir, None, force_get=True)
        return encoders

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        # compile source files, target files & yield sample = {'inputs': input_line, 'targets': target_line}
        raise NotImplementedError

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoders = self.get_or_create_vocab(data_dir, tmp_dir)
        return text_problems.text2text_generate_encoded(sample_generator=generator,
                                                        vocab=encoders["inputs"],
                                                        targets_vocab=encoders["targets"],
                                                        has_inputs=self.has_inputs)

    def generate_text_for_source_vocab(self, data_dir, tmp_dir):
        for i, sample in enumerate(self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
            yield sample["inputs"]
            if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
                break

    def generate_text_for_target_vocab(self, data_dir, tmp_dir):
        for i, sample in enumerate(self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
            yield sample["targets"]
            if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
                break

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        for i, sample in enumerate(self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
            yield sample["inputs"]
            yield sample["targets"]
            if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
                break

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        if force_get:
            vocab_filepath = os.path.join(data_dir, self.vocab_filename)
            encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
            encoders = {"inputs": encoder,
                        "targets": encoder}

        else:
            encoder = generator_utils.get_or_generate_vocab_inner(
                data_dir=data_dir,
                vocab_filename=self.vocab_filename,
                vocab_size=self.approx_vocab_size,
                generator=self.generate_text_for_vocab(data_dir, tmp_dir),
                max_subtoken_length=50)
            encoders = {"inputs": encoder,
                        "targets": encoder}
        return encoders


class ParallelCorpusProblemWithContexts(ParallelCorpusProblem):

    BEGINNING_OF_CONTEXT = "<BOC>"
    BOC_ID = text_encoder.NUM_RESERVED_TOKENS

    def __init__(self, was_reversed=False, was_copy=False):
        super(ParallelCorpusProblemWithContexts, self).__init__(was_reversed=was_reversed, was_copy=was_copy)

        self._num_sentences = 0
        self._feed_concatenated = False

    @property
    def num_sentences(self):
        return self._num_sentences

    @property
    def additional_reserved_tokens(self):
        return [self.BEGINNING_OF_CONTEXT]

    def get_contexts_name(self):
        return ["context" + "_%d" % i for i in range(self.num_sentences)]

    @property
    def feed_concatenated(self):
        """Concatenate context sentences into a single context input
        before feeding into the model. Set by model hyperparameter."""
        return self._feed_concatenated

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        if force_get:
            vocab_filepath = os.path.join(data_dir, self.vocab_filename)
            encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
            encoders = {"inputs": encoder,
                        "targets": encoder}
            for name in self.get_contexts_name():
                encoders[name] = encoder

        else:
            encoder = generator_utils.get_or_generate_vocab_inner(
                data_dir=data_dir,
                vocab_filename=self.vocab_filename,
                vocab_size=self.approx_vocab_size,
                generator=self.generate_text_for_vocab(data_dir, tmp_dir),
                max_subtoken_length=50,
                reserved_tokens=text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens)
            encoders = {"inputs": encoder,
                        "targets": encoder}
            for name in self.get_contexts_name():
                encoders[name] = encoder

        return encoders

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoders = self.get_or_create_vocab(data_dir, tmp_dir)
        return self.text2text_generate_encoded_with_contexts(sample_generator=generator, encoders=encoders)

    def text2text_generate_encoded_with_contexts(self, sample_generator, encoders):
        for sample in sample_generator:
            sample["inputs"] = encoders["inputs"].encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)
            sample["targets"] = encoders["targets"].encode(sample["targets"])
            sample["targets"].append(text_encoder.EOS_ID)

            for name in self.get_contexts_name():
                sample[name] = encoders[name].encode(sample[name])
                sents = sample[name]
                sents.append(text_encoder.EOS_ID)
                sents.insert(0, self.BOC_ID)
                sample[name] = sents

            yield sample

    def hparams(self, defaults, model_hparams):
        (super(ParallelCorpusProblemWithContexts, self).hparams(defaults, model_hparams))
        p = model_hparams
        p.add_hparam("num_context_sentences", self.num_sentences)
        if p.get("feed_concatenated") is not None:
            self._feed_concatenated = p.feed_concatenated

        p = defaults
        if not self.feed_concatenated:
            for name in self.get_contexts_name():
                p.modality[name] = p.modality["inputs"]
                p.vocab_size[name] = p.vocab_size["inputs"]
        else:
            p.modality["context"] = p.modality["inputs"]
            p.vocab_size["context"] = p.vocab_size["inputs"]

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = (
            super(ParallelCorpusProblemWithContexts, self).example_reading_spec())

        for name in self.get_contexts_name():
            data_fields[name] = tf.VarLenFeature(tf.int64)

        return (data_fields, data_items_to_decoders)

    def text_iterator_with_context(self, source_txt_path, target_txt_path):
        with tf.gfile.Open(source_txt_path) as source_file:
            with tf.gfile.Open(target_txt_path) as target_file:
                source_lines = source_file.readlines()
                target_lines = target_file.readlines()
                assert len(source_lines) == len(target_lines)
                features = {}
                context_names = self.get_contexts_name()

                for i in range(len(source_lines)):
                    if i > len(context_names) - 1:
                        features["inputs"] = source_lines[i].strip()
                        features["targets"] = target_lines[i].strip()
                        for j in range(len(context_names)):
                            features[context_names[j]] = source_lines[i - j - 1].strip()
                        yield features
                    else:
                        continue

    def preprocess_example(self, example, mode, hparams):
        """Runtime preprocessing."""
        if self.feed_concatenated:
            context_list = []
            for context_name in self.get_contexts_name():
                context_list.append(example.pop(context_name))
            example["context"] = tf.concat(context_list, 0)

        return super(
            ParallelCorpusProblemWithContexts, self).preprocess_example(
            example, mode, hparams)

    def text2text_generate_encoded_with_randomized_contexts(self, sample_generator, encoders, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            yield
        elif dataset_split == problem.DatasetSplit.EVAL:
            yield

        for sample in sample_generator:
            sample["inputs"] = encoders["inputs"].encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)
            sample["targets"] = encoders["targets"].encode(sample["targets"])
            sample["targets"].append(text_encoder.EOS_ID)

            for name in self.get_contexts_name():
                sample[name] = encoders[name].encode(sample[name])
                sents = sample[name]
                vocab_len = encoders[name].vocab_size
                for i in range(len(sents)):
                    sents[i] = randint(2 + len(self.additional_reserved_tokens), vocab_len-1)
                sents.append(text_encoder.EOS_ID)
                sents.insert(0, self.BOC_ID)
                sample[name] = sents

            yield sample
