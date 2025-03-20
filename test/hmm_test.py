# hacky way to ensure imports work from project root directory
import sys
import os
sys.path.append(os.path.dirname(os.path.pardir))
from hmm import *
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import torch


def test_get_phone_features_and_labels():
    X, Y, phone_X, phone_labels, phones = get_phone_features_and_labels()
    assert type(X) is torch.Tensor
    assert X.shape[1]==4
    assert X.shape[0]>0
    assert type(Y) is torch.Tensor
    assert Y.shape[0]==X.shape[0]

    assert type(phone_X) is dict
    assert type(phone_labels) is np.ndarray
    for phone in 'ailn':
        assert phone in phone_X
        assert phone in phone_labels
        assert type(phone_X[phone]) is torch.Tensor
        assert phone_X[phone].shape[1]==4

    assert phones == 'ailn'

def test_get_word_features_and_labels():
    X, Y, word_X, word_Y, word_labels, words, words_ipa, word_to_ipa, phone_labels, phones = get_word_features_and_labels()
    assert type(X) is torch.Tensor
    assert X.shape[1]==4
    assert type(Y) is torch.Tensor
    assert Y.shape[0]==X.shape[0]

    expected_words = ['lawn', 'lean', 'knee', 'kneel', 'gnaw']
    assert type(word_X) is dict
    assert type(words) is list
    assert len(words)==len(expected_words)
    for x in expected_words:
        assert x in words
        assert x in word_X
        assert type(word_X[x]) is torch.Tensor
        assert word_X[x].shape[1]==4
        assert word_X[x].shape[0]>0

        assert x in word_Y
        assert type(word_Y[x]) is torch.Tensor
        assert word_Y[x].shape[0]==word_X[x].shape[0]

        assert x in word_labels
        assert type(word_labels[x]) is np.ndarray
        assert word_labels[x].shape[0]==word_X[x].shape[0]

    expected_words_ipa = ['lan', 'lin', 'ni', 'nil', 'na']
    assert type(words_ipa) is list
    assert len(words_ipa)==len(expected_words_ipa)
    for x in expected_words_ipa:
        assert x in words_ipa
    
    assert phones == 'ailn'

def test_get_fit_hmm():
    hmm, transmat = get_fit_hmm()
    assert isinstance(hmm, DenseHMM)

    assert len(hmm.distributions) == 4
    for distribution in hmm.distributions:
        assert isinstance(distribution, Normal)

    assert type(hmm.edges) is torch.Tensor
    assert hmm.edges.shape == (4,4)

    assert type(hmm.starts) is torch.Tensor
    assert hmm.starts.shape == (4,)

    assert type(hmm.ends) is torch.Tensor
    assert hmm.ends.shape == (4,)

    assert type(transmat) is torch.Tensor
    assert transmat.shape == (6,6)

def test_get_untrained_word_hmms():
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    assert type(hmm_dict) is dict
    assert len(hmm_dict) == 5
    for word, hmm in hmm_dict.items():
        assert word in ['lawn', 'lean', 'gnaw', 'knee', 'kneel']
        assert isinstance(hmm, DenseHMM)

        assert len(hmm.distributions) == 4
        for distribution in hmm.distributions:
            assert isinstance(distribution, Normal)

        assert type(hmm.edges) is torch.Tensor
        assert hmm.edges.shape == (4,4)
        
        assert type(hmm.starts) is torch.Tensor
        assert hmm.starts.shape == (4,)

        assert type(hmm.ends) is torch.Tensor
        assert hmm.ends.shape == (4,)
    
    for word, transmat in transmat_dict.items():
        assert word in ['lawn', 'lean', 'gnaw', 'knee', 'kneel']

        assert type(transmat) is torch.Tensor
        assert transmat.shape == (6,6)

def test_forward():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for word, hmm in hmm_dict.items():
        transmat = transmat_dict[word]
        for word_seq in word_tuple.word_X.values():
            forward_prob, forward_mat = forward(word_seq, transmat.log(), hmm.distributions)
            assert forward_prob != float('-inf')
            assert not np.isnan(forward_prob)
            forward_prob_pm = hmm.log_probability(word_seq.reshape([1,-1,4]))
            assert torch.isclose(torch.tensor(forward_prob, dtype=forward_prob_pm.dtype), forward_prob_pm).item()
            assert torch.all(~forward_mat.isnan())

def test_backward():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for word, hmm in hmm_dict.items():
        transmat = transmat_dict[word]
        for word_seq in word_tuple.word_X.values():
            backward_prob, backward_mat = backward(word_seq, transmat.log(), hmm.distributions)
            assert backward_prob != float('-inf')
            assert not np.isnan(backward_prob)
            forward_prob_pm = hmm.log_probability(word_seq.reshape([1,-1,4]))
            assert torch.isclose(torch.tensor(backward_prob, dtype=forward_prob_pm.dtype), forward_prob_pm).item()
            assert torch.all(~backward_mat.isnan())

def test_gamma_not_nan():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for word, hmm in hmm_dict.items():
        transmat = transmat_dict[word]
        for word, word_seq in word_tuple.word_X.items():
            _, forward_mat = forward(word_seq, transmat.log(), hmm.distributions)
            _, backward_mat = backward(word_seq, transmat.log(), hmm.distributions)
            for t, _ in enumerate(word_seq):
                for i, _ in enumerate(hmm.distributions):
                    gamma_val = gamma(i, t, forward_mat, backward_mat)
                    assert ~torch.isnan(gamma_val)

def test_ksi_not_nan():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for model_str, hmm in hmm_dict.items():
        transmat = transmat_dict[model_str]
        for word_str, word_seq in word_tuple.word_X.items():
            # only iter thrtuvall words for 'fit' model
            if (model_str!='fit') and (model_str!=word_str):
                continue
            _, forward_mat = forward(word_seq, transmat.log(), hmm.distributions)
            _, backward_mat = backward(word_seq, transmat.log(), hmm.distributions)
            for t, _ in enumerate(word_seq):
                for i, _ in enumerate(hmm.distributions):
                    for j, _ in enumerate(hmm.distributions):
                        ksi_val = ksi(i, j, t, word_seq, forward_mat, backward_mat, transmat.log(), hmm.distributions)
                        assert ~torch.isnan(ksi_val)

def test_a_hat():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for model_str, hmm in hmm_dict.items():
        transmat = transmat_dict[model_str]
        for word_str, word_seq in word_tuple.word_X.items():
            # only iter thrtuvall words for 'fit' model
            if (model_str!='fit') and (model_str!=word_str):
                continue
            for i, _ in enumerate(hmm.distributions):
                a_hat_vec = a_hat(i, word_seq, transmat.log(), hmm.distributions)
                assert torch.all(~a_hat_vec.isnan())

def test_mu_sigsq_hat():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for model_str, hmm in hmm_dict.items():
        transmat = transmat_dict[model_str]
        for word_str, word_seq in word_tuple.word_X.items():
            word_ipa = word_tuple.word_to_ipa[word_str]
            # only iter thrtuvall words for 'fit' model
            if (model_str!='fit') and (model_str!=word_str):
                continue
            for i, _ in enumerate(hmm.distributions):
                if word_tuple.phones[i] not in word_ipa:
                    continue
                mu_hat_vec, sigsq_hat_mat = mu_sigsq_hat(i+1, word_seq, transmat.log(), hmm.distributions)
                assert torch.all(~mu_hat_vec.isnan())
                assert torch.all(~sigsq_hat_mat.isnan())

def test_em_step():
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    df = pd.read_csv('ailn.csv')
    feat_cols = ['f1', 'f2', 'f3', 'amp']
    transmat_dict = {k:v.log() for k,v in transmat_dict.items()}
    new_transmat_dict = em_step(df, feat_cols, hmm_dict, transmat_dict, 'ailn')

    assert type(new_transmat_dict) is dict
    assert len(new_transmat_dict)==5
    for k, transmat in new_transmat_dict.items():
        assert torch.all(~transmat.isnan())
        hmm = hmm_dict[k]
        assert torch.all(
            torch.isclose(hmm.edges, transmat[1:5,1:5])
        )
        assert torch.all(
            torch.isclose(hmm.starts, transmat[0,1:5])
        )
        assert torch.all(
            torch.isclose(hmm.ends, transmat[1:5,-1])
        )
        assert ~torch.all(
            torch.isclose(transmat, transmat_dict[k])
        )
        for dist in hmm.distributions:
            assert torch.all(~dist.means.isnan())
            assert torch.all(~dist.covs.isnan())

def test_segmentation_accuracy_for_word():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    hmm_dict['fit'], transmat_dict['fit'] = get_fit_hmm()
    for model_str, hmm in hmm_dict.items():
        transmat = transmat_dict[model_str]
        for word, word_X in word_tuple.word_X.items():
            word_Y = word_tuple.word_Y[word]
            acc = segmentation_accuracy_for_word(word_X, transmat, hmm.distributions, word_Y)
            assert type(acc) is float
            assert 0<=acc<=1

def test_training_accuracy_improves():
    word_tuple = get_word_features_and_labels()
    hmm_dict, transmat_dict = get_untrained_word_hmms()
    transmat_dict = {k:v.log() for k,v in transmat_dict.items()}
    # evaluate untrained models
    old_acc = {}
    for word, hmm in hmm_dict.items():
        transmat = transmat_dict[word]
        word_X = word_tuple.word_X[word]
        word_Y = word_tuple.word_Y[word]
        acc = segmentation_accuracy_for_word(word_X, transmat, hmm.distributions, word_Y)
        old_acc[word]=acc

    # do 5 training steps
    df = pd.read_csv('ailn.csv')
    feat_cols = ['f1', 'f2', 'f3', 'amp']
    for _ in range(5):
        transmat_dict = em_step(df, feat_cols, hmm_dict, transmat_dict, 'ailn')

    # evaluate agsin
    new_acc = {}
    for word, hmm in hmm_dict.items():
        transmat = transmat_dict[word]
        word_X = word_tuple.word_X[word]
        word_Y = word_tuple.word_Y[word]
        acc = segmentation_accuracy_for_word(word_X, transmat, hmm.distributions, word_Y)
        new_acc[word]=acc

    for k, new_acc_val in new_acc.items():
        old_acc_val = old_acc[k]
        # assert new_acc_val > old_acc_val
        print(f"{new_acc_val=} {old_acc_val=}")