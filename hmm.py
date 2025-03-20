import torch
import numpy as np
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import pandas as pd
from collections import namedtuple
from sklearn.metrics import accuracy_score

# ------------------------ #
# math and iteration utils #
# ------------------------ #

def reversed_enum(a, start=0):
    return reversed(list(enumerate(a, start=start)))

def add_logprobs(log_probs: np.ndarray) -> float:
    if hasattr(log_probs, 'detach'):
        log_probs = log_probs.detach()
    # need quadruple precision to prevent underflow
    if log_probs.dtype is not np.float128:
        log_probs=np.array(log_probs, dtype=np.float128)
    probs=np.exp(log_probs)
    probs_sum=probs.sum()
    logprob_sum=np.log(probs_sum)
    return logprob_sum.astype(float)

# ------------- #
# hmm functions #
# ------------- #

def forward(observations, transition_mat_log, states):
    forward_mat = torch.full((len(states)+2,len(observations)), -torch.inf)
    # initial timestep
    forward_mat[0,0]=0 # always start in initial state
    initial_observation = observations[0]
    for j, state in enumerate(states, start=1):
        transition_logprob=transition_mat_log[0,j]
        emission_logprob=state.log_probability(initial_observation.reshape([1,len(states)]))
        forward_mat[j,0]=transition_logprob+emission_logprob
    # remaining timesteps
    for t, observation in enumerate(observations[1:], start=1):
        for j, curr_state in enumerate(states, start=1):
            emission_logprob = curr_state.log_probability(observation.reshape([1,len(states)])).item()
            logprobs = torch.zeros(len(states))
            for i, _ in enumerate(states, start=1):
                transition_logprob = transition_mat_log[i,j]
                prev_forward = forward_mat[i,t-1]
                logprobs[i-1]=prev_forward+transition_logprob
            logprob=add_logprobs(logprobs)
            logprob+=emission_logprob
            forward_mat[j,t]=logprob
    # transitions to end state
    end_logprobs = torch.zeros(len(states))
    for i, state in enumerate(states, start=1):
        transition_logprob=transition_mat_log[i,-1]
        prev_forward = forward_mat[i,-1]
        end_logprobs[i-1]=transition_logprob+prev_forward
    logprob=add_logprobs(end_logprobs)
    forward_mat[-1,-1]=logprob
    return logprob, forward_mat

def backward(observations, transition_mat_log, states):
    backward_mat = torch.full((len(states)+2,len(observations)), -torch.inf)
    # final timestep
    backward_mat[-1,-1]=0 # always end in final state
    final_observation = observations[-1]
    for i, state in enumerate(states, start=1):
        transition_logprob=transition_mat_log[i,-1]
        emission_logprob=state.log_probability(final_observation.reshape([1,len(states)]))
        backward_mat[i,-1]=transition_logprob+emission_logprob
    # remaining timesteps
    for t, observation in reversed_enum(observations[:-1]):
        for i, curr_state in enumerate(states, start=1):
            emission_logprob = curr_state.log_probability(observation.reshape([1,len(states)])).item()
            logprobs = torch.zeros(len(states))
            for j, _ in enumerate(states, start=1):
                transition_logprob = transition_mat_log[i,j]
                next_backward = backward_mat[j,t+1]
                logprobs[j-1]=next_backward+transition_logprob
            logprob=add_logprobs(logprobs)
            logprob+=emission_logprob
            backward_mat[i,t]=logprob
    # transitions to initial state
    init_logprobs = torch.zeros(len(states))
    for i, state in enumerate(states, start=1):
        transition_logprob=transition_mat_log[0,i]
        next_backward = backward_mat[i,0]
        init_logprobs[i-1]=transition_logprob+next_backward
    logprob=add_logprobs(init_logprobs)
    backward_mat[0,0]=logprob
    return logprob, backward_mat

def viterbi(observations, transition_mat_log, states):
    viterbi_mat = torch.full((len(states)+2,len(observations)), fill_value=-np.inf)
    backtrace = torch.zeros_like(viterbi_mat, dtype=int)

    # initial timestep - same as before
    initial_observation = observations[0]
    for j, state in enumerate(states, start=1):
        transition_logprob=transition_mat_log[0,j]
        emission_logprob=state.log_probability(initial_observation.reshape([1,len(states)]))
        viterbi_mat[j,0]=transition_logprob+emission_logprob

    # remaining timesteps
    for t, observation in enumerate(observations[1:], start=1):
        for j, curr_state in enumerate(states, start=1):
            emission_logprob = curr_state.log_probability(observation.reshape([1,len(states)]))
            prev_viterbi_vec = viterbi_mat[:,t-1]
            transition_vec = transition_mat_log[:,j]
            path_likelihoods = prev_viterbi_vec+transition_vec+emission_logprob

            max_path_likelihood = path_likelihoods.max()
            likely_prev_state = path_likelihoods.argmax() # argmax returns the index of the max value

            viterbi_mat[j,t]=max_path_likelihood
            backtrace[j,t]=likely_prev_state

    # transitions to end state
    final_viterbi_vec = viterbi_mat[:,-1]
    final_transition_vec = transition_mat_log[:,-1]
    final_likelihoods = final_viterbi_vec + final_transition_vec
    max_final_likelihood = final_likelihoods.max()
    likely_prefinal_state = final_likelihoods.argmax()
    
    viterbi_mat[-1,-1]=max_final_likelihood
    backtrace[-1,-1]=likely_prefinal_state
    
    # decode path from backtrace
    prev_state = likely_prefinal_state
    path = torch.zeros(len(observations+2), dtype=int)
    path[-1]=-1
    # so we can iterate thru columns
    backtrace_iter = backtrace.transpose(0,1)
    for t, idcs in reversed_enum(backtrace_iter):
        path[t]=prev_state
        prev_state=idcs[prev_state]
    print(decode_path(path-1))
    return path, viterbi_mat

def decode_path(path, phones='ailn'):
    return ''.join(phones[i] for i in path)

def segmentation_accuracy_for_word(word_observations, transition_mat_log, states, Y):
    path, _ = viterbi(word_observations, transition_mat_log, states)
    # shift indices left
    path-=1
    return accuracy_score(Y, path)

def ksi(i, j, t, observations, forward, backward, transition_mat_log, states):
    """
    i and j in [start, *states, end]
    """
    forward_i = forward[i,t]
    backward_j = backward[j,t]
    transition = transition_mat_log[i,j]
    if (j==len(states)+1) or (j==-1):
        # can't transition to final state before final timestep
        if t<len(observations)-1:
            return float('-inf')
        # else emission probability is 1 (log(1)=0) when transitioning to final state at final timestep
        emission = 0
    else:
        emission = states[j-1].log_probability(observations[t].reshape([1,len(states)])).item()
    
    seq_prob = forward[-1,-1]
    if seq_prob == float('-inf'):
        return float('-inf')

    ksi_val = forward_i + backward_j + transition + emission - seq_prob
    return ksi_val

def gamma(i, t, forward, backward):
    """
    i in [start, *states, end]
    """
    forward_i = forward[i,t]
    backward_i = backward[i,t]
    seq_prob = forward[-1,-1]
    return forward_i + backward_i - seq_prob

def a_hat(i, observations, transition_mat_log, states):
    """
    i and j in [start, *states, end]
    """

    ksi_sums = torch.full((len(states)+2,), float('-inf'))
    _, forward_mat = forward(observations, transition_mat_log, states)
    _, backward_mat = backward(observations, transition_mat_log, states)

    for j in range(1, len(states)+2):
        ksi_j = torch.zeros(len(observations))
        for t in range(len(observations)):
            ksi_log = ksi(i,j,t,observations,forward_mat,backward_mat,transition_mat_log,states)
            ksi_j[t]=ksi_log
        ksi_sums[j]=add_logprobs(ksi_j)

    total_ksi = add_logprobs(ksi_sums)
    if total_ksi == float('-inf'):
        return torch.full((len(states)+2,), float('-inf'))
    a_hat_vec = ksi_sums - total_ksi
    return a_hat_vec

def mu_sigsq_hat(i, observations, transition_mat_log, states):
    """
    i in [start, *states, end]
    """
    _, forward_mat = forward(observations, transition_mat_log, states)
    _, backward_mat = backward(observations, transition_mat_log, states)
    
    # convert to numpy since we'll be using float128
    observations = observations.numpy()
    gamma_vec_log = np.array(
        [gamma(i,t,forward_mat,backward_mat) for t in range(len(observations))],
        dtype=np.float128,
    )
    gamma_vec = np.exp(gamma_vec_log)
    weighted_observations = observations*gamma_vec[:,None]
    mu_hat = weighted_observations.sum(axis=0)/gamma_vec.sum()

    observation_minus_mean = observations-mu_hat
    observation_minus_mean_dot = np.stack([column[:,None]@column[None,:] for column in observation_minus_mean])
    numerator = observation_minus_mean_dot * gamma_vec[:,None,None]
    sigma_hat = numerator.sum(axis=0)/gamma_vec.sum()
    
    sigma_hat = torch.tensor(sigma_hat.astype(np.float64))
    mu_hat = torch.tensor(mu_hat.astype(np.float64))

    return mu_hat, sigma_hat

def em_step(df, feat_cols, hmm_dict, word_transitions_dict, phones):
    num_states = len(list(hmm_dict.values())[0].distributions)
    state_means = torch.zeros((num_states, len(feat_cols)))
    state_covs = torch.zeros((num_states, len(feat_cols), len(feat_cols)))
    new_transitions={}
    for word in df['word'].unique():
        word_mask = df['word']==word
        word_ipa = df.loc[word_mask, 'word_ipa'].iloc[0]
        state_idcs = list(set(phones.index(c)+1 for c in word_ipa))
        word_feats = torch.tensor(df.loc[word_mask, feat_cols].to_numpy())
        word_hmm = hmm_dict[word]
        states = word_hmm.distributions
        word_trans_mat = word_transitions_dict[word]

        new_transition_mat = torch.full_like(word_trans_mat, -torch.inf)
        new_transition_mat[0] = word_trans_mat[0] # initial transition probabilities don't change
        for i in state_idcs:
            # expected transition probabilities
            a_hat_vec = a_hat(i, word_feats, word_trans_mat, states)
            # set transition probs for state i for given word
            new_transition_mat[i]=a_hat_vec

            # collect emission probabilities
            mu_hat_vec, sigmasq_hat_mat = mu_sigsq_hat(i, word_feats, word_trans_mat, states)
            weight_for_avg = len(word_feats)/len(df)
            state_means[i-1]+=mu_hat_vec*weight_for_avg
            state_covs[i-1]+=sigmasq_hat_mat*weight_for_avg
        add_hmm_edges(word_hmm, torch.exp(new_transition_mat), states)
        new_transitions[word]=new_transition_mat
    for i in state_idcs[1:]: # ignore initial state
        states[i-1].means=torch.nn.Parameter(torch.tensor(state_means[i-1]), requires_grad=False)
        states[i-1].covs=torch.nn.Parameter(torch.tensor(state_covs[i-1]), requires_grad=False)
    return new_transitions

# ---------------------- #
# data and model loading #
# ---------------------- #

def get_phone_features_and_labels(
        csv_path='ailn.csv',
        feat_cols=['f1', 'f2', 'f3', 'amp'],
        phones='ailn',
    ):
    df = pd.read_csv(csv_path)
    X = torch.Tensor(df[feat_cols].to_numpy())
    phone_labels = df['phone'].to_numpy()
    Y = torch.tensor(df['phone'].apply(phones.index).to_numpy())
    phone_X = {}
    for phone in phones:
        phone_mask = df['phone']==phone
        phone_idcs = df[phone_mask].index
        phone_X[phone]=X[phone_idcs]

    feature_tuple = namedtuple(
        'phone_features',
        ['X', 'Y', 'phone_X', 'phone_labels', 'phones']
    )
    return feature_tuple(X, Y, phone_X, phone_labels, phones)


def get_word_features_and_labels(
        csv_path='ailn.csv',
        feat_cols=['f1', 'f2', 'f3', 'amp'],
        phones='ailn',
    ):
    df = pd.read_csv(csv_path)
    X = torch.Tensor(df[feat_cols].to_numpy())
    phone_labels = df['phone'].to_numpy()
    Y = torch.tensor(df['phone'].apply(phones.index).to_numpy())
    words = list(df['word'].unique())
    words_ipa = list(df['word_ipa'].unique())
    word_X = {}
    word_Y = {}
    word_to_ipa = {}
    word_labels = {}
    for word in words:
        word_mask = df['word']==word
        word_idcs = df[word_mask].index
        word_X[word]=X[word_idcs]
        word_Y[word]=Y[word_idcs]
        word_labels[word]=phone_labels[word_idcs]
        word_ipa = df.loc[word_mask, 'word_ipa'].iloc[0]
        word_to_ipa[word]=word_ipa
        
    feature_tuple = namedtuple(
        'word_features',
        ['X', 'Y', 'word_X', 'word_Y', 'word_labels', 'words', 'words_ipa', 'word_to_ipa', 'phone_labels', 'phones']
    )
    return feature_tuple(
        X, Y, word_X, word_Y, word_labels, words, words_ipa, word_to_ipa, phone_labels, phones
    )

def add_start_and_end(sequence, num_states=None):
    if type(sequence) is str:
        return '^'+sequence+'$'
    if num_states is None:
        num_states = len(sequence)
    start_i = torch.tensor([num_states])
    end_i = torch.tensor([num_states+1])
    return torch.concat([start_i, sequence, end_i])

def get_transition_matrix(state_sequence, states):
    num_states = len(states)
    transition_counts = torch.zeros((num_states,num_states))
    curr_states=state_sequence[:-1]
    next_states=state_sequence[1:]

    for i, state1 in enumerate(states):
        state1_mask = curr_states==state1
        for j, state2 in enumerate(states[1:], start=1): # never transition into initial state
            state2_mask = next_states==state2
            transition_counts[i,j]=len(curr_states[state1_mask&state2_mask])
    transitions_out = transition_counts.sum(axis=1).reshape((num_states,1))
    transition_mat=transition_counts/transitions_out
    transition_mat[transition_mat.isnan()]=0
    return transition_mat

def add_hmm_edges(hmm, transition_mat, states):
    for i, state1 in enumerate([hmm.start,]+states):
        for j, state2 in enumerate(states+[hmm.end,], start=1):
            weight = transition_mat[i,j]
            if weight==0:
                continue
            if i==0 and j==len(states)+1:
                continue
            hmm.add_edge(state1, state2, weight)
    
def get_fit_hmm():
    """
    Return an HMM-single mixture Gaussian model fit to segmentations from dataset.
    """
    phone_tuple = get_phone_features_and_labels()
    phone_X = phone_tuple.phone_X
    phones = phone_tuple.phones
    hmm = DenseHMM()
    for phone in phones:
        phone_dist = Normal().fit(phone_X[phone])
        hmm.add_distribution(phone_dist)

    word_tuple = get_word_features_and_labels()
    word_Y = word_tuple.word_Y
    state_seq = torch.concat([
        add_start_and_end(word_seq, len(phones))
        for word_seq in word_Y.values()
    ])
    states = add_start_and_end(torch.arange(0,len(phones)))

    transition_mat = get_transition_matrix(state_seq, states)
    add_hmm_edges(hmm, transition_mat, states=hmm.distributions)

    return hmm, transition_mat

def get_untrained_word_hmms():
    """
    Return HMM-single mixture Gaussian model for each word
    """
    phone_tuple = get_phone_features_and_labels()
    X = phone_tuple.X
    phones = phone_tuple.phones
    dists = [Normal().fit(X) for _ in phones]

    hmm_dict = {}
    transmat_dict = {}
    word_tuple = get_word_features_and_labels()
    for word, word_seq in word_tuple.word_Y.items():
        word_seq = add_start_and_end(word_seq, len(phones))
        states = add_start_and_end(torch.arange(0,len(phones)))
        transition_mat = get_transition_matrix(word_seq, states)
        hmm = DenseHMM(dists)
        add_hmm_edges(hmm, transition_mat, states=hmm.distributions)
        hmm_dict[word]=hmm
        transmat_dict[word]=transition_mat
    return hmm_dict, transmat_dict
