import tensorflow as tf
import numpy as np
import harvester_env_continuous
import os
import shutil

np.random.seed(2)
tf.set_random_seed(2)

cwd=os.getcwd()
ep=np.random.uniform(1,2)
env=harvester_env_continuous.car_env(ep)
ACTION_DIM=env.action_dim
ACTION_BOUND=env.actions_bound
STATE_DIM=env.state_dim
# MAX_EPISODES = 500
# MAX_EP_STEPS = 600
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 2000
BATCH_SIZE = 16
VAR_MIN = 0.3
RENDER = True
LOAD = False
DISCRETE_ACTION = False

with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

class Actor():
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess=sess
        self.a_dim=action_dim
        self.a_bound=action_bound
        self.lr=learning_rate
        self.t_replace_iter=t_replace_iter
        self.t_replace_counter=0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)
            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')


    def _build_net(self,s, scope, trainable):
        with tf.variable_scope(scope):
            init_w=tf.random_normal_initializer(mean=0, stddev=0.3, dtype=tf.float32)
            init_b=tf.constant_initializer(0.01, dtype=tf.float32)
            net=tf.layers.dense(s, 30, activation=tf.nn.relu,
                                kernel_initializer=init_w, bias_initializer=init_b,
                               trainable=trainable, name='l1')
            net=tf.layers.dense(net, 20, activation=tf.nn.relu,
                                kernel_initializer=init_w, bias_initializer=init_b,
                                trainable=trainable, name='l2')
            with tf.variable_scope('a'):
                actions=tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh,
                                        kernel_initializer=init_w, bias_initializer=init_b,
                                        trainable=trainable, name='a')
                scaled_a=tf.multiply(actions, self.a_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s.reshape((1,2))  # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

class Critic():
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess=sess
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.lr=learning_rate
        self.gamma=gamma
        self.t_replace_iter=t_replace_iter
        self.t_replace_counter=0

        with tf.variable_scope('Critic'):
            # input (s,a), output q
            self.a=a
            self.q=self._build_net(S, self.a, 'eval_net', True)

            # input (s_,a_), output q_ for q_target
            self.q_=self._build_net(S_, a_, 'target_net', False)

            self.e_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q=R + self.gamma*self.q_

        with tf.variable_scope('TD_error'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.target_q,self.q))

        with tf.variable_scope('C_train'):
            self.train_op=tf.train.RMSPropOptimizer(self.lr).minimize(loss=self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads=tf.gradients(self.q, a)[0]


    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(mean=0, stddev=0.3, dtype=tf.float32)
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

class Memory():
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

sess=tf.Session()

actor=Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A,REPLACE_ITER_A)
critic=Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M=Memory(MEMORY_CAPACITY, dims=2*STATE_DIM+ACTION_DIM+1)

saver=tf.train.Saver()
path=cwd+'/harvester_control_log/'

if LOAD:
    saver.restore(sess,tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

def train():
    var=2.
    for i_ep in range(200):
        # ep = np.random.uniform(1, 2)
        s=env.reset(ep)
        ep_step=0

        while True:
            env.render()
            a=actor.choose_action(s)
            a=np.clip(np.random.normal(a,var), *ACTION_BOUND)
            s_,r,done,breakdown=env.step(a)
            M.store_transition(s,a,r,s_)

            if M.pointer>MEMORY_CAPACITY:
                var=max([var*0.99995,VAR_MIN])
                b_M=M.sample(BATCH_SIZE)
                b_s=b_M[:,:STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r=b_M[:,-STATE_DIM-1:-STATE_DIM]
                b_s_=b_M[:,-STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)

            s=s_
            ep_step += 1
            if done or  breakdown:
                print('Ep:', i_ep,
                      '| Steps: %i' % int(ep_step),
                      '| Explore: %.2f' % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

def eval():
    env.set_fps(30)
    while True:
        s=env.reset()
        while True:
            env.render()
            a=actor.choose_action(s)
            s_,r,done,breakdown=env.step(a)
            s=s_
            if done or breakdown:
                break

if __name__=='__main__':
    if LOAD:
        eval()
    else:
        train()
