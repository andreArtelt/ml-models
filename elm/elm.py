import tensorflow as tf
import numpy as np
from sklearn.base import RegressorMixin
import pickle


class ELM(RegressorMixin):
    def __init__(self, n_hidden=10, n_output=1, act='tanh', l1_penalty=0.0, l2_penalty=0.0001, solver='sgd', loss='mse',
                 verbose=True, verbose_eval=1, max_iter=1000, tol=0.0001, early_stopping=False,
                 reduce_learning_rate_on_plt=True, rl_patience=10, reduce_learning_rate=lambda x: 0.9*x, batch_size=128,
                 shuffle=True, solver_param=dict({'learning_rate': 0.01}), input=None, output=None, sess=None):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.act = act
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.solver = solver
        self.loss = loss
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.reduce_learning_rate_on_plt = reduce_learning_rate_on_plt
        self.rl_patience = rl_patience
        self.reduce_learning_rate = reduce_learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.solver_param = solver_param
        self.input = input
        self.output = output

        self.__ready = False
        self.__sess = sess

    def __check_params(self, X, y, w):
        if self.n_hidden <= 0 or self.n_output <= 0:
            raise ValueError('Number of neurons have to be > 0')
        if self.l1_penalty > 0 and self.solver == 'direct':
            raise ValueError('L1 regularization can not be used with solver \'direct\'')
        if y.shape[1] != self.n_output:
            raise ValueError('Dimension of y does not match dimension of output layer (Expect y.ndim = self.n_output)')
        if X.shape[0] != y.shape[0]:
            raise ValueError('Shape of X does not match shape of y. (Number of rows have to be equal)')
        if w is not None:
            if w.shape[0] != X.shape[0]:
                raise ValueError('Shape of X does not match shape of sample_weight. (Number of rows have to be equal)')
            if w.ndim != 1:
                raise ValueError('Dimension of sample_weight > 1 (Expect 1d array)')
        if self.act not in ['tanh', 'sigmoid', 'relu']:
            raise ValueError('Unknown activation function (Expect a value from [\'tanh\', \'sigmoid\', \'relu\'] )')
        if self.solver not in ['direct', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'momentum']:
            raise ValueError('Unknown solver (Expect a value from [\'direct\', \'sgd\', \'rmsprop\', \'adadelta\', \'adagrad\', \'adam\', \'momentum\'])')
        if self.loss not in ['mse', 'rmse', 'mae', 'hinge', 'log']:
            raise ValueError('Unkown loss (Expect value from [\'mse\', \'rmse\', \'mae\', \'hinge\', \'log\'])')
        if self.max_iter < 0:
            raise ValueError('max_iter have to be >= 0')
        if self.batch_size < 0:
            raise ValueError('batch_size have to be > 0')
        if self.solver is not 'direct' and self.solver_param is None:
            raise ValueError('No arguments for solver given (If you use a different solver than \'direct\' you have to provide some additional arguments like learning_rate, etc.)')
        if self.solver is 'direct' and self.loss is not 'mse':
            raise ValueError('Direct solver work with \'mse\' loss only!')

    def __apply_act(self, t):
        if self.act == 'relu':
            return tf.nn.relu(t)
        if self.act == 'tanh':
            return tf.nn.tanh(t)
        if self.act == 'sigmoid':
            return tf.nn.sigmoid(t)

    def __get_loss(self, y, y_pred):
        if self.loss == 'mse':
            return tf.losses.mean_squared_error(y, y_pred)
        if self.loss == 'rmse':
            return tf.sqrt(tf.losses.mean_squared_error(y, y_pred))
        if self.loss == 'mae':
            return tf.reduce_mean(tf.losses.absolute_difference(y, y_pred))
        if self.loss == 'hinge':
            return tf.losses.hinge_loss(y, y_pred)
        if self.loss == 'log':
            return tf.losses.log_loss(y, y_pred)

    def __get_optimizer(self):
        solver_param = self.solver_param.copy()  # Learning rate will be in a custom variable
        solver_param.pop('learning_rate')

        if self.solver == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, **solver_param)
        if self.solver == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, **solver_param)
        if self.solver == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, **solver_param)
        if self.solver == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate, **solver_param)
        if self.solver == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate, **solver_param)
        if self.solver == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate, **solver_param)

    def load(self, path):
        X = None
        y = None
        with open(path + '.shape', 'rb') as shape_file:
            data = pickle.load(shape_file)
            X = np.empty(data['X'])
            y = np.empty(data['Y'])
            self.n_hidden = data['n_hidden']
            self.n_output = data['n_output']
            self.act = data['act']
            self.l1_penalty = data['l1_penalty']
            self.l2_penalty = data['l2_penalty']
            self.loss = data['loss']

        self.__build(X, y)

        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

        tf.train.Saver().restore(self.__sess, './' + path)
        self.__ready = True

    def save(self, path):
        tf.train.Saver().save(self.__sess, path)
        with open(path + '.shape', 'wb') as shape_file:
            x_shape = self.__x.get_shape().as_list()
            x_shape[0] = 1

            y_shape = self.__y_true.get_shape().as_list()
            y_shape[0] = 1

            pickle.dump(dict({'X': x_shape, 'Y': y_shape, 'n_hidden': self.n_hidden, 'n_output': self.n_output,
                              'act': self.act, 'l1_penalty': self.l1_penalty, 'l2_penalty': self.l2_penalty,
                              'loss': self.loss}),
                        shape_file)

    def build_ex(self, X):
        if self.input is None:
            self.__x = tf.placeholder(tf.float32, (None, X.shape[1]))
        else:
            self.__x = self.input
        if self.output is None:
            self.__y_true = tf.placeholder(tf.float32, (None, self.n_output))
        else:
            self.__y_true = self.output

        self.w_1 = tf.Variable(tf.truncated_normal([self.__x.get_shape().as_list()[1], self.n_hidden]),
                          trainable=False, name='W1')
        self.b_1 = tf.Variable(tf.truncated_normal([self.n_hidden]), trainable=False, name="b1")
        self.w_2 = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_output]), trainable=True, name='W2')
        self.b_2 = tf.Variable(tf.truncated_normal([self.n_output]), trainable=True, name="b2")

        if self.n_output == self.__x.get_shape().as_list()[1]:  # Can be used as an autoencoder?
            self.x_ae = self.__apply_act(tf.matmul(self.__x + self.b_2, tf.transpose(self.w_2)))

        self.y_1 = self.__apply_act(tf.matmul(self.__x, self.w_1) + self.b_1)
        self.__y_pred = tf.matmul(self.y_1, self.w_2) + self.b_2

        self.__loss = self.__get_loss(self.__y_true, self.__y_pred) + self.l1_penalty * tf.reduce_sum(tf.abs(self.w_2)) +\
                      self.l2_penalty * tf.reduce_sum(tf.square(self.w_2))

        self.learning_rate = tf.placeholder(tf.float32, ())

        if self.solver == 'direct':
            self.w_2_opt = tf.matmul(
                tf.matmul(
                    tf.matrix_inverse(
                        tf.matmul(tf.transpose(self.y_1), self.y_1) +
                        self.l2_penalty * tf.Variable(tf.eye(self.n_hidden), trainable=False)
                    ), tf.transpose(self.y_1)
                ), self.__y_true)
            self.b_2_opt = tf.reduce_mean(self.__y_true, 0) - tf.reduce_mean(tf.matmul(self.y_1, self.w_2), 0)
        else:
            self.opt = self.__get_optimizer().minimize(self.__loss, var_list=[self.w_2, self.b_2])

    def build(self, X, sample_weight=None):
        if self.__ready:
            return

        # Build model
        self.build_ex(X)

        # Init/Create session if necessary
        if self.__sess is None:
            self.__sess = tf.Session()
            self.__sess.run(tf.global_variables_initializer())

        self.__ready = True

    def fit(self, X, y, sample_weight=None, feed_dict=None):
        # Check params
        self.__check_params(X, y, sample_weight)

        # Build model
        self.build(X, sample_weight)

        # Fit
        if feed_dict is None:
            feed_dict = {self.__x: X, self.__y_true: y}

        if self.solver == 'direct':
            w_opt = self.__sess.run(self.w_2_opt, feed_dict=feed_dict)
            self.__sess.run(self.w_2.assign(w_opt))
            b_opt = self.__sess.run(self.b_2_opt, feed_dict=feed_dict)
            self.__sess.run(self.b_2.assign(b_opt))

            loss_opt = self.__sess.run(self.__loss, feed_dict=feed_dict)

            if self.verbose:
                print "Final loss: " + str(loss_opt)
        else:
            self.fit_steps(X, y, self.max_iter, solver_param=self.solver_param, sample_weight=sample_weight,
                           feed_dict=feed_dict)

        self.__ready = True

        return self

    def fit_steps(self, X, y, n_max_steps=10, solver_param=dict({'learning_rate': 0.001}), sample_weight=None,
                  feed_dict=None):
        if self.solver == 'direct':
            raise Exception('Step wise fitting is supported for iterative solvers only!')

        # Build model
        self.build(X, sample_weight)

        # Create optimizer
        self.max_iter = n_max_steps
        self.solver_param = solver_param

        # Setup learning rate
        learning_rate = 0.001
        if 'learning_rate' in self.solver_param:
            learning_rate = self.solver_param['learning_rate']
        rl_counter = 0

        # Epochs/Rounds
        old_loss = float('inf')
        for i in range(self.max_iter):
            loss_opt = float('inf')

            if self.shuffle:
                idx = range(0, X.shape[0])
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]

            # Batches
            for b in range(0, X.shape[0], self.batch_size):
                x_train = X[b : b + self.batch_size]
                y_train = y[b : b + self.batch_size]

                if feed_dict is None:
                    feed_dict = dict({self.__x: x_train, self.__y_true: y_train, self.learning_rate: learning_rate})
                    values = feed_dict
                else:
                    values = dict({feed_dict['x']: x_train, feed_dict['y']: y_train, self.learning_rate: learning_rate})

                _, loss_opt = self.__sess.run([self.opt, self.__loss], feed_dict=values)

            if self.verbose is True and i % self.verbose_eval == 0:
                print "Itr " + str(i) + " " + str(loss_opt)

            # Early stopping
            if self.early_stopping is True and np.abs(old_loss - loss_opt) < self.tol:
                break

            # No improvement but an worsening => reduce learning rate
            if self.reduce_learning_rate_on_plt:
                if old_loss < loss_opt and old_loss != float('inf'):
                    if rl_counter >= self.rl_patience:
                        learning_rate = self.reduce_learning_rate(learning_rate)
                        rl_counter = 0
                        if self.verbose:
                            print "Reduce learning rate to " + str(learning_rate)
                    else:
                        rl_counter += 1

            old_loss = loss_opt

        if self.verbose:
            print "Final loss: " + str(loss_opt)

        return self

    def transform_ae(self, X, feed_dict=None):
        if self.n_output != X.shape[1]:
            raise Exception('This model is not an autoencoder! (n_output != n_input)')

        self.build(X)

        if feed_dict is None:
            feed_dict = {self.__x: X}

        return self.__sess.run(self.x_ae, feed_dict=feed_dict)

    def score(self, X, y, sample_weight=None, feed_dict=None):
        self.build(X, sample_weight)

        if feed_dict is None:
            feed_dict = {self.__x: X, self.__y_true: y}

        return self.__sess.run(self.__loss, feed_dict=feed_dict)

    def predict(self, x, feed_dict=None):
        if self.__ready is False:
            raise Exception('You have to fit the model first!')

        if feed_dict is None:
            feed_dict = {self.__x: x}

        return self.__sess.run(self.__y_pred, feed_dict=feed_dict)
