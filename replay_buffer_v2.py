import numpy as np
import random

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    # assume the x adv etc are sampled in batches.
    def add(self, x, adv, recon_x, loss,mu,var,att):
            # print(x.shape)
            # print(adv.shape)
            # print(recon_x.shape)
            # print(loss.shape)
            # print(mu.shape)
            # print(var.shape)
            # print()
        for i in range(x.shape[0]):

            data = (x[i], adv[i], recon_x[i], loss[i],mu[i],var[i],att)

            if self._next_idx >= len(self._storage): # increasing size
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data

            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        xs, advs, recon_xs, losses,mus,vars,att_types= [], [], [], [],[],[],[]
        for i in idxes:
            data = self._storage[i]
            x, adv, recon_x, loss,mu,var,att = data
            xs.append(np.array(x, copy=False))
            advs.append(np.array(adv, copy=False))
            recon_xs.append(recon_x)
            losses.append(np.array(loss, copy=False))
            mus.append(mu)
            vars.append(var)
            att_types.append(att)

        return np.array(xs), np.array(advs), np.array(recon_xs), np.array(losses),np.array(mus),np.array(vars),np.array(att_types)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, x, adv, recon_x, loss,mu,var,att_type):
        """See ReplayBuffer.store_effect"""

        for i in range(x.shape[0]):
            idx = self._next_idx
            data = (x[i], adv[i], recon_x[i], loss[i], mu[i], var[i],att_type)
            #print(data[3])
            if self._next_idx >= len(self._storage):  # increasing size
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data

            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
            # is this scaled?
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities,losses_new):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert len(idxes) == len(losses_new)
        for i,(idx, priority )in enumerate(zip(idxes, priorities)):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            (x1,x2,x3,x4,x5,x6,x7) = self._storage[idx]
            self._storage[idx] = (x1,x2,x3,losses_new[i],x5,x6,x7) # update loss
            self._max_priority = max(self._max_priority, priority)
