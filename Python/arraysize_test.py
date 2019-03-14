
from tempfile import TemporaryFile
import numpy as np
import pickle, os, time
import gzip, lzma, bz2, h5py

outfile = TemporaryFile()
x = np.zeros([84,84,1,1,1,10**6//4],dtype=np.int8)
#np.save(outfile, x)
print(f"a Numpy array of size {x.shape} would be {x.nbytes/10**9} Gigabytes")