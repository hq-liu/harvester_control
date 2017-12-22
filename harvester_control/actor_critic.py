
import tensorflow as tf
import numpy as np
import harvester_env_continuous

np.random.seed(2)
tf.set_random_seed(2)

LR_A=0.001
LR_C=0.01
GAMMA=0.9

class Actor():
    def __init__(self, sess, n_features, action_bound):
        self.sess=sess

        self.action_bound=action_bound
        self.s=tf.placeholder(dtype=tf.float32,shape=[1,n_features],name='state')
        self.a=tf.placeholder(dtype=tf.float32,shape=None,name='action')
        self.td_error=tf.placeholder(tf.float32,shape=None,name='td_error')

        init_w=tf.random_normal_initializer(mean=0, stddev=0.3, dtype=tf.float32)
        init_b=tf.constant_initializer(0.01, dtype=tf.float32)

        with tf.variable_scope('Actor'):
            l1=tf.layers.dense(
                inputs=self.s, units=20, activation=tf.nn.relu,
                kernel_initializer=init_w, bias_initializer=init_b,
                name='l1'
            )

            mu = tf.layers.dense(
                inputs=l1,
                units=1,  # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu'
            )

            sigma = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.softplus,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(1.),  # biases
                name='sigma'
            )

        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.variable_scope('exp_v'):
            log_prob=self.normal_dist.log_prob(self.a)
            self.exp_v=log_prob*self.td_error
            self.exp_v+= 0.01*self.normal_dist.entropy()

        with tf.variable_scope('train'):
            self.train_op=tf.train.AdamOptimizer(learning_rate=LR_A).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis,:]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis,:]
        return self.sess.run(self.action, {self.s: s})



class Critic():
    def __init__(self, sess, n_features):
        self.sess=sess

        self.s=tf.placeholder(dtype=tf.float32, shape=[1,n_features],name='state')
        self.v_=tf.placeholder(tf.float32, shape=[1,1], name='v_next')
        self.r=tf.placeholder(tf.float32, shape=None, name='r')

        init_w = tf.random_normal_initializer(mean=0, stddev=0.3, dtype=tf.float32)
        init_b = tf.constant_initializer(0.01, dtype=tf.float32)
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=init_w,  # weights
                bias_initializer=init_b,  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=init_w,  # weights
                bias_initializer=init_b,  # biases
                name='V'
            )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR_C).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

OUTPUT_GRAPH=False
sess=tf.Session()
ep=np.random.uniform(1,2)
env=harvester_env_continuous.car_env(ep)
actor=Actor(sess, n_features=2, action_bound=env.actions_bound)
critic=Critic(sess, n_features=2)

sess.run(tf.global_variables_initializer())
if OUTPUT_GRAPH:
    tf.summary.FileWriter('logs_AC/',sess.graph)

for i_ep in range(300):
    ep = np.random.uniform(1, 2)
    s = env.reset(ep)
    t=0
    ep_rs=[]
    while True:

        env.render()
        a=actor.choose_action(s)

        s_,r,done,breakdown=env.step(a)

        td_error=critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s=s_
        t += 1
        ep_rs.append(r)
        if done or breakdown:
            ep_rs_sum=sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            print("episode:", i_ep, "  reward:", int(running_reward))
            break



