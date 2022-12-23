import itertools
import operator
import re
import tensorflow as tf
from keras import layers, losses
from keras.saving.object_registration import register_keras_serializable
from keras.utils.control_flow_util import smart_cond
from keras.utils.metrics_utils import sparse_categorical_matches
from keras.utils.losses_utils import compute_weighted_loss, ReductionV2 as Reduction
from keras.utils.tf_utils import shape_type_conversion
from tensorflow.python.distribute import distribution_strategy_context


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class HSMax(layers.Layer):
    def __init__(self, tree, label_smoothing=0., loss_reduction=Reduction.AUTO, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        classes = self._validate_node(tree, root=True)
        self.input_spec = [layers.InputSpec(ndim=2, axes={-1: classes}), layers.InputSpec(ndim=1, dtype='int64')]

        self.tree = tree
        self.loss_reduction = loss_reduction
        self.label_smoothing = label_smoothing

    def _validate_node(self, node, root=False):
        if not isinstance(node, dict):
            raise ValueError(f'Wrong node type: {type(node)}. Expecting "dict".')

        required_keys = {'id', 'total'}
        missed_keys = required_keys - set(node.keys())
        if missed_keys:
            raise ValueError(f'Node misses required keys: {missed_keys}.')

        allowed_keys = {'cover', 'child'}
        disallowed_keys = set(node.keys()) - required_keys - allowed_keys
        if disallowed_keys:
            raise ValueError(f'Node contains disallowed keys: {disallowed_keys}.')

        child = node.get('child', [])
        if 1 == len(child):
            raise ValueError('Node shoud not contain single child.')

        if root:
            if 'cover' in node:
                raise ValueError('Root node should not contain "cover" key.')

            if len(child) < 2:
                raise ValueError('Root node should contain at least 2 child nodes.')

            cover = self._build_cover(node)
            if len(cover) != len(set(cover)):
                raise ValueError('All id\'s and covers must be unique.')

        return len(child) + sum([self._validate_node(c) for c in child])

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[0][-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self._feature_slices, self._labels_lookup = {}, {}
        last, queue = 0, [self.tree]
        while queue:
            oov_check = 0 == last
            node = queue.pop(0)

            child = node.get('child', [])
            if not child:
                continue

            queue.extend(child)

            index = node['id']
            lname = re.sub('[^0-9a-z_]', '_', f'lookup_{index}'.lower())

            self._feature_slices[index] = (last, last + len(child))
            self._labels_lookup[index] = lname
            setattr(self, lname, TTL([self._build_cover(c) for c in child], oov_check, name=lname))

            last += len(child)

            super().build(input_shape)

    def _build_cover(self, node):
        cover, queue = [], [node]
        while queue:
            node = queue.pop(0)
            cover.append(node['id'])
            cover.extend(node.get('cover', []))

            child = node.get('child', [])
            if not child:
                continue

            queue.extend(child)

        return cover

    def call(self, inputs, *args, **kwargs):
        logits, labels = inputs

        losses, accuracies, sizes = [], [], []
        root, queue = True, [self.tree]
        while queue:
            node = queue.pop(0)

            child = node.get('child', [])
            if not child:
                continue

            queue.extend(child)

            index = node['id']
            start, stop = self._feature_slices[index]
            targets = getattr(self, self._labels_lookup[index])(labels)

            if root:
                root, size = False, 1
                loss, accuracy = self._compute_node(logits, targets, start, stop, None)
            else:
                mask = targets != -1
                size, loss, accuracy = smart_cond(
                    tf.reduce_any(mask),
                    lambda: (1, *self._compute_node(logits, targets, start, stop, mask)),
                    lambda: (0, 0., 0.))

            losses.append(loss)
            accuracies.append(accuracy)
            sizes.append(size)

        size = tf.cast(sum(sizes), 'float32')

        loss = sum(losses) / size
        self.add_loss(loss)

        accuracy = sum(accuracies) / size
        self.add_metric(accuracy, name='accuracy')

        return logits

    def _compute_node(self, logits, targets, start, stop, mask):
        logits = logits[:, start:stop]
        if mask is not None:
            logits, targets = logits[mask], targets[mask]

        loss = self._compute_loss(logits, targets)
        accuracy = self._compute_accuracy(logits, targets)

        return loss, accuracy

    def _compute_loss(self, logits, targets, weights=None):
        if distribution_strategy_context.has_strategy() and \
                self.loss_reduction in {Reduction.AUTO, Reduction.SUM_OVER_BATCH_SIZE}:
            raise ValueError(
                'Use `Reduction.SUM` or `Reduction.NONE` for loss reduction when  losses are used with '
                '`tf.distribute.Strategy` outside of the built-in training loops. Please see '
                'https://www.tensorflow.org/tutorials/distribute/custom_training for more details.')

        logits = tf.cast(logits, 'float32')

        if self.label_smoothing > 0.:
            targets = tf.one_hot(targets, logits.shape[-1], dtype='float32')
            loss = losses.categorical_crossentropy(
                targets, logits, from_logits=True, label_smoothing=self.label_smoothing)
        else:
            loss = losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)

        loss = compute_weighted_loss(loss, sample_weight=weights, reduction=self.loss_reduction)

        return loss

    def _compute_accuracy(self, logits, targets):
        accuracy = sparse_categorical_matches(targets, logits)
        accuracy = tf.reduce_mean(accuracy)

        return accuracy

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'tree': self.tree,
            'label_smoothing': self.label_smoothing,
            'loss_reduction': self.loss_reduction
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class TTL(layers.IntegerLookup):
    """ Tree target lookup """

    def __init__(self, vocabulary, oov_check=True, **kwargs):
        kwargs.pop('max_values', None)
        kwargs.pop('mask_value', None)
        kwargs.pop('oov_value', None)
        super().__init__(
            max_tokens=None, num_oov_indices=0, mask_token=None, oov_token=-1, vocabulary=vocabulary,
            vocabulary_dtype='int64', idf_weights=None, invert=False, output_mode='int', sparse=False,
            pad_to_max_tokens=False, has_input_vocabulary=True, **kwargs)
        self.oov_check = oov_check

    def adapt(self, data, batch_size=None, steps=None):
        raise NotImplementedError()

    def set_vocabulary(self, vocabulary, idf_weights=None):
        if idf_weights is not None:
            raise ValueError('Setting vocabulary from IDF weights not supported.')

        if isinstance(vocabulary, str):
            raise ValueError('Setting vocabulary from file not supported.')

        if tf.is_tensor(vocabulary):
            raise ValueError('Setting vocabulary from tensor not supported.')

        tokens = list(itertools.chain.from_iterable(vocabulary))
        if self.oov_token in tokens:
            raise ValueError(f'Vocabulary contains OOV token.')

        repeated_tokens = self._find_repeated_tokens(tokens)
        if repeated_tokens:
            raise ValueError(f'Vocabulary has at least one repeated term: {repeated_tokens}.')

        self.lookup_table = self._lookup_table_from_tokens(vocabulary)

    def get_vocabulary(self, include_special_tokens=True):
        del include_special_tokens

        if 0 == self.lookup_table.size():
            return []

        keys, values = self.lookup_table.export()
        keys = self._tensor_vocab_to_numpy(keys).tolist()
        values = values.numpy()

        vocab = sorted(zip(keys, values), key=operator.itemgetter(0))
        vocab = sorted(vocab, key=operator.itemgetter(1))
        vocab = itertools.groupby(vocab, operator.itemgetter(1))
        vocab = [list(map(operator.itemgetter(0), v)) for _, v in vocab]

        return vocab

    def _lookup_table_from_tokens(self, tokens):
        with tf.init_scope():
            keys = list(itertools.chain.from_iterable(tokens))

            values = [[i] * len(t) for i, t in enumerate(tokens)]
            values = list(itertools.chain.from_iterable(values))

            init = tf.lookup.KeyValueTensorInitializer(keys, values, self._key_dtype, self._value_dtype)
            table = tf.lookup.StaticHashTable(init, self._default_value)

            return table

    def _lookup_table_from_file(self, filename):
        raise NotImplementedError()

    def _lookup_dense(self, inputs):
        if not self.oov_check:
            self.num_oov_indices = 1

        result = super()._lookup_dense(inputs)
        self.num_oov_indices = 0

        return result

    def get_config(self):
        config = super().get_config()
        config.update({'oov_check': self.oov_check})

        del config['max_tokens']
        del config['num_oov_indices']
        del config['mask_token']
        del config['oov_token']
        del config['vocabulary_dtype']
        del config['idf_weights']
        del config['invert']

        del config['output_mode']
        del config['sparse']
        del config['pad_to_max_tokens']

        return config
