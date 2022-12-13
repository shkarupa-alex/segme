import os
import json

def tree_21k():
    with open(os.path.join(os.path.dirname(__file__), 'tree21k.json'), 'rb') as f:
        return json.load(f)