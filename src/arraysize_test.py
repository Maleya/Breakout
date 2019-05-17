
#from tempfile import TemporaryFile
import numpy as np
import pickle, os, time
import gzip, lzma, bz2, h5py

#outfile = TemporaryFile()
# x = np.zeros([84,84,1,1,1,10**6//2],dtype=np.int8)
r = np.random.randint(5, size=(84,84,1,1,1,10**6//8))
print(f"a Numpy array of size {r.shape} would be {r.nbytes/10**9} Gigabytes")



# normal save
# np.save('outfile', r)


# gzip 
f = gzip.GzipFile('del_me_gzipfile.npy.gz', "w")
np.save(f, r)
f.close()
# 