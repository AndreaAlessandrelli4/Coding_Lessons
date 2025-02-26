import numpy as np
import tensorflow as tf

class Hopfield_Network:

    def __init__(self):
        self.N = None; self.K = None; self.J = None; self.σ = None;
    
    def prepare(self, J):
        self.N = np.shape(J)[1]; 
        self.J = J

    def dynamics(self, σ0, β, updates, mode="parallel"):
        steps, N, M = np.shape(σ0); J = self.J
        σup = tf.convert_to_tensor(np.copy(σ0),dtype=tf.float32)

        for _ in range(updates):
            h = tf.einsum('Sij,SjA->SiA',J,σup)
            u = np.random.uniform(-1,1,(steps, N, M))
            σup = tf.sign(tf.tanh(β * h) + u)
            self.σ = σup


class TAM3_Network:

    def __init__(self):

        self.N = None; self.K = None; self.J = None; self.L = None; self.σ = None
    
    def prepare(self, J):
        self.L = np.shape(J)[1]
        self.N = np.shape(J)[2]
        self.J = np.copy(J)
    
    def compute_fields(self, input_field, g, ρ):

        J = self.J; N = self.N; σ = self.σ

        h1 = tf.einsum('SABij,SvBj->SvABi',J,σ) # l == b, k == a
        # temp0 = tf.einsum('sABi,sAi->sABi',h1, σ) # sigma_b dot h_a
        temp1 = tf.einsum('AB, B->AB',g, np.sqrt(1.0+ρ))
        temp = tf.einsum('AB, A->AB',temp1,np.sqrt(1.0+ρ))
        ht = tf.einsum('SvABi,AB->SvAi',h1,temp)
        
        return ht

    def dynamics(self, σr, updates, ρ, g_att, β):

        L = self.L
        g1, g2, g3 = g_att
        g = np.zeros((L,L))
        g[0,1] = g[1,0] = g1
        g[0,2] = g[2,0] = g2
        g[1,2] = g[2,1] = g3
        g = tf.convert_to_tensor(g,dtype=tf.float32)
        σr = tf.convert_to_tensor(σr,dtype=tf.float32)
        S, s, L, N  = σr.shape
        self.σ = np.copy(σr)

        for _ in range(updates):
            ht = self.compute_fields(self.σ, g, ρ)
            u = tf.random.uniform(shape=(S,s,L,N))*2-1
            self.σ = tf.sign(tf.tanh(β * ht *self.σ)*self.σ +u)

