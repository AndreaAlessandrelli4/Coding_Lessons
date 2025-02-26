import numpy as np
import tensorflow as tf

def gen_patterns(N, P, L, steps):
    ξ = 2*np.random.randint(0,2,(steps, L, P, N))-1;
    return tf.convert_to_tensor(ξ,dtype=tf.float32)

def gen_patterns_hop(N, P, steps):
    ξ = 2*np.random.randint(0,2,(steps,  P, N))-1;
    return tf.convert_to_tensor(ξ,dtype=tf.float32)


def gen_archetypes(ξ, M, ρ):
    ρ = np.array(ρ)
    r = np.sqrt(1/(ρ*M+1))
    η = [];
    steps, L, P, N =np.shape(ξ)
    for μ in range(P):
        probs = np.random.uniform(size=(steps,L,M,N));
        χ1 = np.einsum('L, sLMN->sLMN', 0.5*(1+r), np.ones((steps,L,M,N)))
        χ = np.sign(χ1 -probs);
        ημ = np.array(tf.einsum('SLai,SLi->SLai',χ,ξ[:,:,μ,:]))
        ημ = np.einsum('aLb,L->aLb',np.mean(ημ,axis=2),1/(r*(1.0+ρ)))
        η.append(ημ)
    η = tf.convert_to_tensor(η,dtype=tf.float32)
    return η, r


def gen_archetypes_Hop(ξ, M, ρ):
    if ρ ==0:
        r = 1.0
    else:
        r = np.sqrt(1/(ρ*M+1))
    η = [];
    steps, P, N =np.shape(ξ)
    for μ in range(P):
        probs = np.random.uniform(size=(steps,M,N));
        χ = np.sign(0.5*(1+r)-probs);
        ημ = np.array(tf.einsum('Sai,Si->Sai',χ,ξ[:,μ,:]))
        ημ = np.mean(ημ,axis=2)/(r*(1.0+ρ))
        η.append(ημ)
    η = tf.convert_to_tensor(η,dtype=tf.float32)
    return η, r

def gen_dataset(ξ, r, M):
    r = np.array(r)
    steps, L, P, N  = np.shape(ξ)
    probs = np.random.uniform(size=(steps,L,M,N));
    χ1 = np.einsum('L, sLMN->sLMN', 0.5*(1+r), np.ones((steps,L,M,N)))
    χ = np.sign(χ1 -probs);
    ημ = np.array(tf.einsum('SLai,SLi->SLai',χ,ξ[:,:,0,:]))
    ημ = tf.convert_to_tensor(ημ,dtype=tf.float32)
    return ημ

def gen_dataset_hop(ξ, r, M):
    steps, P, N  = np.shape(ξ)
    probs = np.random.uniform(size=(steps,M,N));
    χ = np.sign(0.5*(1+r)-probs);
    ημ = np.array(tf.einsum('Sai,Si->Sai',χ,ξ[:,0,:]))
    ημ = tf.convert_to_tensor(ημ,dtype=tf.float32)
    return ημ

def unsupervised_J(η, M):
    N = np.shape(η)[1];
    return tf.einsum('ai,aj->ij',η,η)/(N*M)



def Hebb_J(η):
    P, steps, L, N = np.shape(η);
    return tf.einsum('kSAi,kSBj->SABij',η,η)/(N)

def Hebb_J_hop(η):
    steps, P, N = np.shape(η);
    return tf.einsum('Ski,Skj->Sij',η,η)/(N)