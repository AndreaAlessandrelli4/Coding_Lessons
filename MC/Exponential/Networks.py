import numpy as np
import tensorflow as tf

class Exp_Network:

    def __init__(self):
        self.N = None; self.K = None; self.iter = None; self.J = None; self.σ = None; self.model = None; self.P=None;
    
    def prepare(self, J, model, P):
        self.iter, self.K, self.N = J.shape
        self.J = J
        self.model = model
        self.P = P

    def dynamics(self, σ0, updates):
        model = self.model
        K  = self.K; N  = self.N; J = self.J; P = self.P
        σ = tf.convert_to_tensor(np.copy(σ0),dtype=tf.float32)
        if model.lower() =='hopfield':
            for _ in range(updates):
                h1 = tf.einsum('aki,aLi->aLk',J,σ)/N
                h = tf.einsum('aLk, aki -> aLi',h1**(P-1),J)
                σ = tf.sign(h)
                self.σ = σ
        else:
            for _ in range(updates):
                h0 = tf.exp(-tf.einsum('aki,aLi->aLki',J,σ))
                h1 = tf.einsum('aki,aLi->aLk',J, σ)/N -1.0
                h2 = tf.einsum('aki, aLki -> aLki',J, h0)
                h = tf.einsum('aLk, aLki -> aLi',tf.exp(h1*N),h2)
                σ = tf.sign(h)
                self.σ = σ
