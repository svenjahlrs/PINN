
#sys.path.insert(0, '../Utilities/')

import tensorflow as tf
import numpy as np
from scipy.interpolate import griddata
from pyDOE import lhs
from libs.plotting import *
from csv import writer
from libs.data_preprocessing_LWT import *
import time
from os.path import exists

np.random.seed(1234)
tf.set_random_seed(1234)

def write_csv_line(path: str, line):
    with open(path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(line)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, dic, layers_phi, layers_eta, lb, ub, name_save):
        '''

        :param xt_0: x and t values for initial condition shape: (N_0 x 2)
        :param eta_0: corresponding true solution eta(x, t)
        :param xt_lb: collocation points at lower boundary for periodic boundary conditions, shape (N_pb x 2)
        :param xt_ub: collocation points at lower boundary for periodic boundary conditions, shape (N_pb x 2)
        :param xtz_b12: x, t and z collocation points for boundary conditions at z=0, shape (N_b12 x 3)
        :param xtz_b3: x, t and z collocation for boundary conditions at z=-d, shape (N_b3 x 3)
        :param xtz_f: x, t and z collocation points in entire domain for PDE-residual, shape (N_f x 3)
        :param layers_phi: list of neurons per layer for phi-network
        :param layers_eta: list of neurons per layer for eta-network
        :param lb: [x_min, t_min, z_min]
        :param ub: [x_max, t_max, z_max]
        '''

        self.best_loss = None
        self.epoch = 0
        self.start_time = time.time()
        self.name_save = name_save

        self.optimizer = None
        self.lb = lb  # lower boundary [X_min, T_min, Z_min] of domain
        self.ub = ub  # upper boundary [X_max, T_max, Z_max] of domain

        # supervised training data
        self.x_0 = dic['xt_0'][:, 0:1]
        self.t_0 = dic['xt_0'][:, 1:2]
        self.eta_0 = dic['eta_0']

        # collocation points at boundaries (x=x_min, t, z=0) for periodic boundary conditions
        self.x_lb_phi = dic['xtz_lb_phi'][:, 0:1]
        self.t_lb_phi = dic['xtz_lb_phi'][:, 1:2]
        self.z_lb_phi = dic['xtz_lb_phi'][:, 2:3]
        self.x_ub_phi = dic['xtz_ub_phi'][:, 0:1]
        self.t_ub_phi = dic['xtz_ub_phi'][:, 1:2]
        self.z_ub_phi = dic['xtz_ub_phi'][:, 2:3]

        self.x_lb_eta = dic['xt_lb_eta'][:, 0:1]
        self.t_lb_eta = dic['xt_lb_eta'][:, 1:2]
        self.x_ub_eta = dic['xt_ub_eta'][:, 0:1]
        self.t_ub_eta = dic['xt_ub_eta'][:, 1:2]

        # collocation points at boundary (x, t, z=0) for free surface boundary conditions
        self.x_bc12 =dic['xtz_bc12'][:, 0:1]
        self.t_bc12 = dic['xtz_bc12'][:, 1:2]
        self.z_bc12 = dic['xtz_bc12'][:, 2:3]

        # collocation points at boundary (x, t, z=-d) for bed boundary condition
        self.x_bc3 = dic['xtz_bc3'][:, 0:1]
        self.t_bc3 = dic['xtz_bc3'][:, 1:2]
        self.z_bc3 = dic['xtz_bc3'][:, 2:3]

        # collocation points inside entire (x, t, z) domain for pde-residual
        self.x_pde = dic['xtz_pde'][:, 0:1]
        self.t_pde = dic['xtz_pde'][:, 1:2]
        self.z_pde = dic['xtz_pde'][:, 2:3]


        # Initialize NNs
        self.layers_phi = layers_phi  # number of nodes each layer [3, .., 1]
        self.layers_eta = layers_eta  # number of nodes each layer [2, .., 1]
        self.weights_phi, self.biases_phi = self.initialize_NN(layers_phi)  # list of weight and bias tensors each node
        self.weights_eta, self.biases_eta = self.initialize_NN(layers_eta)  # list of weight and bias tensors each node

        # tf Placeholders of shape (none x 1)
        self.x_0_tf = tf.placeholder(tf.float32, shape=[None, self.x_0.shape[1]])
        self.t_0_tf = tf.placeholder(tf.float32, shape=[None, self.t_0.shape[1]])

        self.eta_0_tf = tf.placeholder(tf.float32, shape=[None, self.eta_0.shape[1]])

        self.x_lb_eta_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb_eta.shape[1]])
        self.t_lb_eta_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb_eta.shape[1]])

        self.x_ub_eta_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub_eta.shape[1]])
        self.t_ub_eta_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub_eta.shape[1]])

        self.x_lb_phi_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb_phi.shape[1]])
        self.t_lb_phi_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb_phi.shape[1]])
        self.z_lb_phi_tf = tf.placeholder(tf.float32, shape=[None, self.z_lb_phi.shape[1]])

        self.x_ub_phi_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub_phi.shape[1]])
        self.t_ub_phi_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub_phi.shape[1]])
        self.z_ub_phi_tf = tf.placeholder(tf.float32, shape=[None, self.z_ub_phi.shape[1]])

        self.x_bc12_tf = tf.placeholder(tf.float32, shape=[None, self.x_bc12.shape[1]])
        self.t_bc12_tf = tf.placeholder(tf.float32, shape=[None, self.t_bc12.shape[1]])
        self.z_bc12_tf = tf.placeholder(tf.float32, shape=[None, self.z_bc12.shape[1]])

        self.x_bc3_tf = tf.placeholder(tf.float32, shape=[None, self.x_bc3.shape[1]])
        self.t_bc3_tf = tf.placeholder(tf.float32, shape=[None, self.t_bc3.shape[1]])
        self.z_bc3_tf = tf.placeholder(tf.float32, shape=[None, self.z_bc3.shape[1]])

        self.x_pde_tf = tf.placeholder(tf.float32, shape=[None, self.x_pde.shape[1]])
        self.t_pde_tf = tf.placeholder(tf.float32, shape=[None, self.t_pde.shape[1]])
        self.z_pde_tf = tf.placeholder(tf.float32, shape=[None, self.z_pde.shape[1]])

        self.tf_dict = {self.x_0_tf: self.x_0, self.t_0_tf: self.t_0,
                       self.eta_0_tf: self.eta_0,
                       self.x_lb_eta_tf: self.x_lb_eta, self.t_lb_eta_tf: self.t_lb_eta,
                       self.x_ub_eta_tf: self.x_ub_eta, self.t_ub_eta_tf: self.t_ub_eta,
                       self.x_lb_phi_tf: self.x_lb_phi, self.t_lb_phi_tf: self.t_lb_phi, self.z_lb_phi_tf: self.z_lb_phi,
                       self.x_ub_phi_tf: self.x_ub_phi, self.t_ub_phi_tf: self.t_ub_phi, self.z_ub_phi_tf: self.z_ub_phi,
                       self.x_bc12_tf: self.x_bc12, self.t_bc12_tf: self.t_bc12, self.z_bc12_tf: self.z_bc12,
                       self.x_bc3_tf: self.x_bc3, self.t_bc3_tf: self.t_bc3, self.z_bc3_tf: self.z_bc3,
                       self.x_pde_tf: self.x_pde, self.t_pde_tf: self.t_pde, self.z_pde_tf: self.z_pde}




        # tf Graphs
        self.eta_0_pred, _ = self.net_eta(x=self.x_0_tf, t=self.t_0_tf)  # prediction for xt_0

        self.eta_lb_pred, _ = self.net_eta(x=self.x_lb_eta_tf, t=self.t_lb_eta_tf)  # prediction for xt_0
        self.eta_ub_pred, _ = self.net_eta(x=self.x_ub_eta_tf, t=self.t_ub_eta_tf)  # prediction for xt_0

        self.eta_b12_pred, self.eta_t_b12_pred = self.net_eta(x=self.x_bc12_tf, t=self.t_bc12_tf)  # prediction for xtz_b12

        self.phi_lb_pred, _, _, _, _ = self.net_phi(x=self.x_lb_phi_tf, t=self.t_lb_phi_tf, z=self.z_lb_phi_tf)
        self.phi_ub_pred, _, _, _, _ = self.net_phi(x=self.x_ub_phi_tf, t=self.t_ub_phi_tf, z=self.z_ub_phi_tf)

        _, self.phi_t_b12_pred, self.phi_z_b12_pred, _, _ = self.net_phi(x=self.x_bc12_tf, t=self.t_bc12_tf, z=self.z_bc12_tf)

        _, _, self.phi_z_b3_pred, _, _ = self.net_phi(x=self.x_bc3_tf, t=self.t_bc3_tf, z=self.z_bc3_tf)  # prediction for xtz_b3

        _, _, _, self.phi_xx_f_pred, self.phi_zz_f_pred = self.net_phi(x=self.x_pde_tf, t=self.t_pde_tf, z=self.z_pde_tf)  # prediction for xtz_f


        # Loss (MSE_0= MSE(eta_true, eta_pred) + MSE_pb_dirichlet + MSE_pb_newman + MSE_b1 + MSE_b2 + MSE_b3 + MSE_f
        self.loss_MSE_0 = tf.reduce_mean(tf.square(self.eta_0_tf - self.eta_0_pred))
        self.loss_MSE_pb_eta = tf.reduce_mean(tf.square(self.eta_lb_pred - self.eta_ub_pred))
        self.loss_MSE_pb_phi = tf.reduce_mean(tf.square(self.phi_lb_pred - self.phi_ub_pred))
        self.loss_MSE_bc1 = tf.reduce_mean(tf.square(self.eta_t_b12_pred - self.phi_z_b12_pred))
        self.loss_MSE_bc2 = tf.reduce_mean(tf.square(self.phi_t_b12_pred + 9.81 * self.eta_b12_pred))
        self.loss_MSE_bc3 = tf.reduce_mean(tf.square(self.phi_z_b3_pred))
        self.loss_MSE_pde = tf.reduce_mean(tf.square(self.phi_xx_f_pred + self.phi_zz_f_pred))

        # Loss 9
        self.loss = 500 * self.loss_MSE_0 + 8 * self.loss_MSE_pb_eta + 8 * self.loss_MSE_pb_phi + 15 * self.loss_MSE_bc1 + 15 * self.loss_MSE_bc2 + 5 * self.loss_MSE_bc3 + 15 * self.loss_MSE_pde


        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        '''
        initalizes all weights and biases
        :param layers: array of nodes per layer
        :return: list of wight tensor for each layer, list of bias tensor for each layer
        '''
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # weights tensor with Xavier std
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                            dtype=tf.float32)  # initialize biases with zeros
            weights.append(W)  # list of initialized weight tensor for each layer
            biases.append(b)  # list of initialized biases
        return weights, biases

    def xavier_init(self, size):
        '''
        initializes weights according to Xavier initialization depending on number of incoming and outgoing connections
        :param size: [nodes of previous layer, nodes of next layer]
        :return: tf Variable of shape: (nodes of previous layer x nodes of next layer) filled with random values
        '''
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        if weights[0].shape[0] == 2:
            lb = self.lb[0:2]
            ub = self.ub[0:2]
        elif weights[0].shape[0] == 3:
            lb = self.lb
            ub = self.ub

        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_eta(self, x, t):

        X = tf.concat([x, t], 1)  # two input nodes (None x 2)

        eta = self.neural_net(X, self.weights_eta, self.biases_eta)  # two output nodes (None x 2)
        # eta_x = tf.gradients(eta, x)[0]  # partial differentiation
        eta_t = tf.gradients(eta, t)[0]  # partial differentiation

        return eta, eta_t

    def net_phi(self, x, t, z):

        X = tf.concat([x, t, z], 1)  # three input nodes (None x 3)

        phi = self.neural_net(X, self.weights_phi, self.biases_phi)  # two output nodes (None x 2)

        phi_x = tf.gradients(phi, x)[0]  # partial differentiation
        phi_t = tf.gradients(phi, t)[0]  # partial differentiation
        phi_z = tf.gradients(phi, z)[0]  # partial differentiation

        phi_xx = tf.gradients(phi_x, x)[0]  # partial differentiation
        phi_zz = tf.gradients(phi_z, z)[0]  # partial differentiation

        return phi, phi_t, phi_z, phi_xx, phi_zz

    def callback(self):

        elapsed_time = time.time() - self.start_time

        keys = ['time', 'epoch', 'loss', 'loss_MSE_0', 'loss_MSE_pb_eta', 'loss_MSE_pb_phi', 'loss_MSE_bc1', 'loss_MSE_bc2', 'loss_MSE_bc3', 'loss_MSE_pde']

        vals = [np.round(elapsed_time, 3), self.epoch] + self.sess.run([self.loss, self.loss_MSE_0, self.loss_MSE_pb_eta, self.loss_MSE_pb_phi,
                self.loss_MSE_bc1, self.loss_MSE_bc2, self.loss_MSE_bc3, self.loss_MSE_pde], self.tf_dict)

        print('time:', np.round(elapsed_time, 3), "".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))

        path_loss = f"errors/loss_{self.name_save}.csv"
        if not exists(path_loss):
            write_csv_line(path=path_loss, line=keys)

        write_csv_line(path=path_loss, line=vals)

        if vals[2] < self.best_loss:
            self.best_loss = vals[2]

        self.epoch += 1
        self.start_time = time.time()

    def callback_LBFGS(self, loss, loss_MSE_0, loss_MSE_pb_eta, loss_MSE_pb_phi, loss_MSE_bc1, loss_MSE_bc2, loss_MSE_bc3,loss_MSE_pde):
        elapsed_time = time.time() - self.start_time

        keys = ['time', 'epoch', 'loss', 'loss_MSE_0', 'loss_MSE_pb_eta', 'loss_MSE_pb_phi', 'loss_MSE_bc1', 'loss_MSE_bc2', 'loss_MSE_bc3', 'loss_MSE_pde']

        vals = [np.round(elapsed_time, 3), self.epoch, loss, loss_MSE_0, loss_MSE_pb_eta, loss_MSE_pb_phi, loss_MSE_bc1, loss_MSE_bc2, loss_MSE_bc3, loss_MSE_pde]

        print('time:', np.round(elapsed_time, 3), "".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))

        path_loss = f"errors/loss_{self.name_save}.csv"
        if not exists(path_loss):
            write_csv_line(path=path_loss, line=keys)

        write_csv_line(path=path_loss, line=vals)

        if loss < self.best_loss:
            self.best_loss = loss

        self.epoch += 1
        self.start_time = time.time()

    def train(self, epochsAdam, epochsBFGS):

        self.best_loss = self.sess.run(self.loss, self.tf_dict)

        # Optimizer 1: Adam

        for it in range(epochsAdam):
            self.sess.run(self.train_op_Adam, self.tf_dict)
            self.callback()

        # Optimizer 2:  BFGS
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': epochsBFGS,  # e.g. 50000 in Raissis tutorial,
                                                                         'maxfun': epochsBFGS,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer.minimize(self.sess,
                                feed_dict=self.tf_dict,
                                fetches=[self.loss, self.loss_MSE_0, self.loss_MSE_pb_eta, self.loss_MSE_pb_phi, self.loss_MSE_bc1, self.loss_MSE_bc2, self.loss_MSE_bc3, self.loss_MSE_pde],
                                loss_callback=self.callback_LBFGS)

    def predict(self, x, t):

        tf_dict = {self.x_0_tf: x, self.t_0_tf: t}

        eta_star = self.sess.run(self.eta_0_pred, tf_dict)

        # tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        #
        # f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        # f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return eta_star


if __name__ == "__main__":
    layers_phi = [3, 100, 100, 100, 100, 1]
    layers_eta = [2, 100, 100, 100, 100,  1]

    d = 2  # water depth

    epochsLBFGS = 5000
    epochsAdam = 1000

    delta = 0.1
    datas = ['regular_wave', 'regular_wave_half_WL', 'regular_wave_sin', 'superimposed_wave', 'three_component_superimposed_wave_x4']
    modes = ['snapshots', 'sparse']

    info_path = f'errors/info_delta_{delta}_epochs_{epochsAdam}_{epochsLBFGS}_regular_wave.csv'
    write_csv_line(path=info_path, line=['data', 'mode', 'traintime', 'epochsLBFGS(real)', 'SSP(surf)', 'Loss(best)'])

    for data in datas:
        for mode in modes:

            print(f'\n\n\n\n\n data = {data} mode = {mode} \n \n\n \n \n')

            np.random.seed(1234)
            tf.set_random_seed(1234)

            name_save = f'epochs_{epochsAdam}_{epochsLBFGS}_data_{data}_mode_{mode}'

            path_loss = f"errors/loss_{name_save}.csv"

            dat = np.load(f'../../Data/superimposed_waves/{data}.npz')

            t = dat['t'][:, None]
            x = dat['x'][:, None]
            Exact = dat['eta']

            X, T = np.meshgrid(x, t)  # meshgrid for NN solution
            # domain bounds
            lb = np.array([np.min(X), np.min(T), -d])  # lower boundary: [X_min, T_min, Z_min]
            ub = np.array([np.max(X), np.max(T), 0])  # upper boundary: [X_max, T_max, Z_max]

            X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

            dict_collpoints = return_collocation_points(X=X, T=T, Exact=Exact, d=d, delta=delta, mode=mode)

            model = PhysicsInformedNN(dict_collpoints,
                                      layers_phi, layers_eta, lb, ub, name_save=name_save)

            start_time = time.time()
            model.train(epochsAdam=epochsAdam, epochsBFGS=epochsLBFGS)
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))


            eta_pred = model.predict(X_star[:, 0:1], X_star[:, 1:2])

            Eta_pred = griddata(X_star, eta_pred.flatten(), (X, T), method='cubic')

            plotting_PINN3(x=x, t=t, Eta_true=Exact, Eta_pred=Eta_pred, xt_train=dict_collpoints['xt_0'],
                           t_is=[0, 50, 100, 150, 200, 250],
                           name_save=name_save)
            plotting_losscurve(path_loss=path_loss, name_save='loss_' + name_save, xmax=epochsAdam + epochsLBFGS,
                               ymax=0.3)

            SSP = SSP_2D(Exact, Eta_pred)
            MSE = np.mean(np.square(Exact - Eta_pred))

            file = open(path_loss)
            epochsLBFGS_real = len(file.readlines()) -epochsAdam

            line = [str(data), str(mode), np.round(elapsed,1), epochsLBFGS_real, np.round(SSP, 4), model.best_loss]
            write_csv_line(path=info_path, line=line)


            del model
