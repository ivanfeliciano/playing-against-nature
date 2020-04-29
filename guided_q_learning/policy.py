import numpy as np

class Policy(object):
    def __init__(self, eps_max, eps_min, eps_test, nb_steps, env, causal=False, ratio=0.5):
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_test = eps_test
        self.nb_steps = nb_steps
        self.env = env
        self.step = 0
        self.causal = causal

    def get_current_value(self, training=True):
        if training:
            a = -float(self.eps_max - self.eps_min) / float(self.nb_steps)
            b = float(self.eps_max)
            value = max(self.eps_min, a * float(self.step) + b)
        else:
            value = self.eps_test
        self.step = (self.step +  1) % self.nb_steps
        # self.step += 1
        return value
    def select_action(self, state, Q, training=True):
        raise NotImplementedError

class EpsilonGreedy(Policy):
    def select_action(self, state, Q, training=True, prob_explore=0.0):
        r = np.random.uniform()
        eps = self.get_current_value(training)
        if r > eps:
            return np.argmax(Q[state])
        r = np.random.uniform()
        if not self.causal:
            return self.env.sample_action()
        if r > 0.8:
            return self.env.sample_action()
        goal = self.env.get_goal()
        macro_state = self.env.get_state()
        targets = []
        for i in range(len(goal)):
            if goal[i] == 1 and macro_state[i] == 0:
                targets.append(i + self.env.num)
        np.random.shuffle(targets)
        for target in targets:
            actions = self.env.causal_structure.get_causes(target)
            if len(actions) > 0:
                return actions.pop()      
        return self.env.sample_action()
