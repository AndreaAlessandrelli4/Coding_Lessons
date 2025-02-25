import numpy as np
import tensorflow as tf

def gen_patterns(N, P, steps):
    ξ = 2*np.random.randint(0,2,(steps, P, N))-1;
    return tf.convert_to_tensor(ξ,dtype=tf.float32)

def gen_dataset(ξ, r, M):
    steps, P, N  = ξ.shape
    probs = np.random.uniform(size=(steps,M,N));
    χ = np.sign(0.5*(1+r)-probs);
    ημ = np.array(tf.einsum('Sai,Si->Sai',χ,ξ[:,0,:]))
    ημ = tf.convert_to_tensor(ημ,dtype=tf.float32)
    return ημ