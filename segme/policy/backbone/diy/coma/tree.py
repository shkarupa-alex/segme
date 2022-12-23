import os
import json
import tensorflow as tf


def tree_21k1k(synsets=False):
    tree_path = os.path.join(os.path.dirname(__file__), 'tree21k1k.json')
    with tf.io.gfile.GFile(tree_path, 'rb') as f:
        tree = json.load(f)

    if synsets:
        return tree

    syns = synsets_21k1k()

    def _syn2id(node, root=False):
        node['id'] = -1 if root else syns.index(node['id'])

        if 'cover' in node:
            node['cover'] = [syns.index(c) for c in node['cover']]

        if 'child' in node:
            node['child'] = [_syn2id(c) for c in node['child']]

        return node

    return _syn2id(tree, root=True)


def synsets_21k1k():
    syns_path = os.path.join(os.path.dirname(__file__), 'synset1k21k.txt')
    with tf.io.gfile.GFile(syns_path, 'r') as f:
        syns = f.read().strip().splitlines()

    return syns
