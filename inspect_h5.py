import sys
sys.path.insert(0, r'D:\python_packages')
import h5py

def print_keys(g, level=0):
    for k in g.keys():
        print('  '*level + k)
        if isinstance(g[k], h5py.Group):
            print_keys(g[k], level+1)

f = h5py.File('models/ann_model.h5', 'r')
print_keys(f['model_weights'])
