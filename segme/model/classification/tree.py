import os
import json
import tensorflow as tf


def heinsen_tree_paths():
    syns = synsets_1k_21k()
    tree = _tree_21k()
    cmap = tree_class_map()

    paths = []
    queue = [(n, []) for n in tree['child']]
    while queue:
        node, parents = queue.pop(0)

        oldidx = syns.index(node['id'])
        newid = cmap[oldidx]
        paths.append(parents + [newid])

        queue.extend([(c, parents + [newid]) for c in node.get('child', [])])

    return paths


def tree_class_map():
    syns = synsets_1k_21k()
    tree = _tree_21k()

    cmap, queue, last = {}, tree['child'], 0
    while queue:
        node = queue.pop(0)

        oldidx = syns.index(node['id'])
        cmap[oldidx] = last

        for cover in node.get('cover', []):
            oldidx = syns.index(cover)
            cmap[oldidx] = last

        queue.extend(node.get('child', []))

        last += 1

    return cmap


def synsets_1k_21k():
    # take first 1000 values to match default imagenet 1k classes
    syns_path = os.path.join(os.path.dirname(__file__), 'synset1k21k.txt')
    with tf.io.gfile.GFile(syns_path, 'r') as f:
        syns = f.read().strip().splitlines()

    return syns


def _tree_21k():
    tree_path = os.path.join(os.path.dirname(__file__), 'tree21k1k.json')
    with tf.io.gfile.GFile(tree_path, 'rb') as f:
        tree = json.load(f)

    return tree