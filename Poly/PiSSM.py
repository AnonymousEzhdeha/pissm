

import tensorflow as tf
from tensorflow import keras as k
import numpy as np

from PiSSMTransitionCell import PiSSMTransitionCell, pack_input, unpack_state, pack_state
from GINSmoothCell import PiSSMSmoothingCell


class PiSSM(k.models.Model):

    def __init__(self, observation_shape, latent_observation_dim, latent_state_dim, output_dim, num_basis, 
                 trans_net_hidden_units=[], never_invalid=False, cell_type="lstm", Qnetwork="Xmlp", USE_CONV = False, Smoothing = False,
                 USE_MLP_AFTER_KGGRU = False,
                 KG_Units = 50,
                 Xgru_Units =15,
                 Fgru_Units =15,
                 KG_InputSize = 100,
                 Xgru_InputSize = 15,
                 Fgru_InputSize = 15,
                 lr = 0.001,
                 lr_decay = 0.5,
                 lr_decay_it = 15,
                result_path = "/media/green/58FA6D84FA6D5F6E/Science and University/ICLR_Revision/iclr_git/Codes/Polybox image imputation/results"):
        """
        obs_shape: shape of the observation 
        obs_dim: latent observation dimension 
        out_dim: dimensionality of model output
        num_basis: number of basis matrices 
        trans_net_hidden_units: hidden units for dynamics network
        never_invalid: boolean indicating whether all observations are available or a part of it is missing
        cell_type: type of cell to use "gin", "lstm" or "gru" 
        Qnetwork: defines the type of inference for Q matrix. "Xmlp", "Xgru", "Fmlp", "Fgru" and "nothing". 
            "Xmlp": Q = MLP(X^+)
            "Xgru": Q = GRU(X^+)
            "Fmlp": Q = MLP(F)
            "Fgru": Q = GRU(F)
            "nothing": Q is learned jointly with the transition matrix (F(Q) in the paper)
        USE_CONV: defines whether use the convolutional layer for the covariance matrix or not
        """
        super().__init__()

        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        self._lsd = latent_state_dim
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        self._ld_output = np.isscalar(self._output_dim)
        self.Smoothing = Smoothing
        self.cell_type = cell_type
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_it = lr_decay_it
        self.result_path = result_path
        # build encoder
        self._enc_hidden_layers = self._time_distribute_layers(self.build_encoder_hidden())

        # we need to ensure the bias is initialized with non-zero values to ensure the normalization does not produce
        # nan
        self._layer_w_mean = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=k.activations.linear,
                           bias_initializer=k.initializers.RandomNormal(stddev=0.05)))
        self._layer_w_mean_norm = k.layers.TimeDistributed(k.layers.Lambda(
            lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        self._layer_w_covar = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=lambda x: k.activations.elu(x) + 1))

        # build transition
        if self.cell_type.lower() == "gin":
            self._cell = PiSSMTransitionCell(self._lsd, self._lod,
                                           number_of_basis=num_basis,
                                           init_kf_matrices=0.05,
                                           init_Q_matrices = 0.05,
                                           init_KF_matrices = 0.1,
                                           Qnetwork = Qnetwork,
                                           USE_CONV = USE_CONV,
                                           trans_net_hidden_units=trans_net_hidden_units,
                                           never_invalid=never_invalid,
                                           USE_MLP_AFTER_KGGRU = USE_MLP_AFTER_KGGRU,
                                            KG_Units = KG_Units,
                                            Xgru_Units =Xgru_Units,
                                            Fgru_Units =Fgru_Units,
                                            KG_InputSize = KG_InputSize,
                                            Xgru_InputSize = Xgru_InputSize,
                                            Fgru_InputSize = Fgru_InputSize,
                                            cell_type="lstm")
        elif self.cell_type.lower() == "lstm":
            print("Running LSTM Baseline")
            self._cell = k.layers.LSTMCell(2 * self._lsd)
        elif self.cell_type.lower() == "gru":
            print("Running GRU Baseline")
            self._cell = k.layers.GRUCell(2 * self._lsd)
        elif self.cell_type.lower() == "encdec":
            print("Running EncDec Baseline")
        else:
            raise AssertionError("Invalid Cell type, needs tp be 'gin', 'lstm', 'gru' or 'encdec'")

        if (self.cell_type.lower() != "encdec"):
            self._layer_rkn = k.layers.RNN(self._cell, return_sequences=True)
            
        if self.Smoothing:
            self._smoothing_cell = PiSSMSmoothingCell(self._lsd,
                                                    self._lod,
                                                    init_kf_matrices = 0.05,
                                                    init_KF_matrices = 0.05,
                                                    USE_CONV = USE_CONV)
            self._layer_smooth = k.layers.RNN(self._smoothing_cell, return_sequences=True)

        self._dec_hidden = self._time_distribute_layers(self.build_decoder_hidden())
        if self._ld_output:
            # build decoder mean
            self._layer_dec_out = k.layers.TimeDistributed(k.layers.Dense(units=self._output_dim))

            # build decoder variance
            self._var_dec_hidden = self._time_distribute_layers(self.build_var_decoder_hidden())
            self._layer_var_dec_out = k.layers.TimeDistributed(
                k.layers.Dense(units=self._output_dim, activation=lambda x: k.activations.elu(x) + 1))

        else:
            self._layer_dec_out = k.layers.TimeDistributed(
                k.layers.Conv2DTranspose(self._output_dim[-1], kernel_size=3, padding="same",
                                         activation=k.activations.sigmoid))


    def build_encoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def build_decoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """
        inputs: original observations
        training: required by k.models.Models
        mask: required by k.models.Model
        
        """
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            img_inputs, obs_valid = inputs
        else:
            assert self._never_invalid, "If invalid inputs are possible, obs_valid mask needs to be provided"
            img_inputs = inputs
            obs_valid = tf.ones([tf.shape(img_inputs)[0], tf.shape(img_inputs)[1], 1])

        # encoder
        enc_last_hidden = self._prop_through_layers(img_inputs, self._enc_hidden_layers)
        w_mean = self._layer_w_mean_norm(self._layer_w_mean(enc_last_hidden))
        w_covar = self._layer_w_covar(enc_last_hidden)

        # log softmax of sampling from DynamicsNet (for reinforce)
        logp_list = []
        # transition
        rkn_in = pack_input(w_mean, w_covar, obs_valid)
        

        if self.cell_type.lower() == 'gin':
            z = self._layer_rkn(rkn_in)
            # unpack outputs;[ post_mean = mu_t|t, (posterior_mean, i.e. mean filtered),
                            # post_covar = sigma_t|t, (posterior_covar, i.e. covar filtered) 
                            # prior_mean = mu_t|t-1 = A_t mu_t-1|t-1, 
                            # prior_covar = sigma_t|t-1 = A_t sigma_t-1|t-1 A_t^T + Q_t
                            # transition_matrix = A_t
            
            post_mean, post_covar, prior_mean, prior_covar, transition_matrix, logp_list = z
            post_covar = tf.concat(post_covar, -1)
            if self.Smoothing:
                smooth_mean_init, smooth_covar_init, post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse, transition_matrix_reverse = self.z_time_reverse(z)
                init_state = pack_state(smooth_mean_init, smooth_covar_init)
                post_mean_reverse, post_covar_reverse = self._layer_smooth((post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse,
                                                            transition_matrix_reverse), initial_state = init_state)
                post_mean_reverse = tf.concat([tf.expand_dims(smooth_mean_init, axis=1), post_mean_reverse], axis=1)
                post_covar_reverse = tf.concat([tf.expand_dims(smooth_covar_init, axis=1), post_covar_reverse], axis=1)
                post_mean = tf.reverse(post_mean_reverse, axis=[1])
                post_covar = tf.reverse(post_covar_reverse, axis =[1])
                post_covar = tf.concat(post_covar, -1)

        elif(self.cell_type.lower() == 'gru' or self.cell_type.lower() == 'lstm'):
            z = self._layer_rkn(rkn_in)
            post_mean, post_covar = unpack_state(z, self._lsd)
            post_covar = tf.concat(post_covar, -1)

        else:
            post_mean = w_mean
            post_covar = w_covar

        # decode
        pred_mean = self._layer_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
        if self._ld_output:
            pred_var = self._layer_var_dec_out(self._prop_through_layers(post_covar, self._var_dec_hidden))
            return tf.concat([pred_mean, pred_var], -1), logp_list
        else:
            return pred_mean, logp_list

    def z_time_reverse(self, z):
        post_mean, post_covar, prior_mean, prior_covar, transition_matrix, _ = z
        smooth_mean_init = post_mean[:, -1, :]
        smooth_covar_init = post_covar[:, -1, :]

        post_mean_reverse = tf.reverse(post_mean[:, :-1, :], [1])
        post_covar_reverse = tf.reverse(post_covar[:, :-1, :], [1])
        prior_mean_reverse = tf.reverse(prior_mean[:, 1:, :], [1])
        prior_covar_reverse = tf.reverse(prior_covar[:, 1:, :], [1])
        transition_matrix_reverse = tf.reverse(transition_matrix[:, 1:, :, :], [1])

        return smooth_mean_init, smooth_covar_init, post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse, transition_matrix_reverse
    # loss functions
    def gaussian_nll(self, target, pred_mean_var):
        """
        output with gaussian assumption distribution
        target: ground truth
        pred_mean_var: mean and covar 
        
        """
        pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1) # [batch, T]
        base_loss = tf.reduce_mean(tf.reduce_mean(sample_wise_error, axis=1)) #scalar 
        # base_loss = tf.reduce_mean(sample_wise_error) #scalar 
        return base_loss

    
    def reinforce_loss(self, target, pred_mean_var, logp_list):
        """
        output reinforce+basement loss
        target: ground truth
        pred_mean_var: mean and covar 
        calculate reinforce+baseline
        
        """
        pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1) # [batch, T]
        base_loss = tf.reduce_mean(tf.reduce_sum(sample_wise_error, axis=1)) #scalar 

        reward = -sample_wise_error 
        ###
        ### Baseline+ REINFORCE
        baseline = tf.reduce_mean(reward, axis=0)  # shape [T]
        baseline = tf.expand_dims(baseline, axis=0)  # [1, T]
        baseline = tf.tile(baseline, [pred_mean.shape[0], 1])  # [batch, T]
        logps = tf.squeeze(logp_list, axis=-1) 

        reward = tf.stop_gradient(reward)
        baseline = tf.stop_gradient(baseline)
        reinforce_term = - (reward - baseline) * logps  # shape [batch, T]
        reinforce_loss = tf.reduce_mean(reinforce_term)  # scalar

        return reinforce_loss

    def rmse(self, target, pred_mean_var):
        """
        MSE
        target: ground truth 
        pred_mean_var: mean and covar 
        
        """
        pred_mean = pred_mean_var[..., :self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean - target) ** 2))

    def bernoulli_nll(self, targets, predictions, uint8_targets=True):
        """ 
        output with bernoulli assumption distribution
        targets: ground truth
        predictions: model's prediction
        uint8_targets: if true it is assumed that the targets are given in uint8 
        
        """
        if uint8_targets:
            targets = targets / 255.0
        point_wise_error = - (
                    targets * tf.math.log(predictions + 1e-12) + (1 - targets) * tf.math.log(1 - predictions + 1e-12))
        red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_wise_error)
    
    def enforce_iss_condition(self, cell_type="gru"):
        """
        Enforce ISS stability constraints on recurrent cell parameters.
        Supports GRU (for now).
        """
        if cell_type == "gru":
            # Grab the GRUCell (for KG, Q, or main transition cell)
            gru_cell = self._cell.KGCELL   # or QCELL depending on context

            # GRU weights:
            # - kernel: [input_dim, 3*units]
            # - recurrent_kernel: [units, 3*units]
            # - bias: [2, 3*units] in TF2 (input + recurrent bias)
            W = None
            U = None
            b = None
            for w in gru_cell.weights:
                if "recurrent_kernel" in w.name:
                    # U = w  # shape [units, 4*units]
                    U = next(w for w in gru_cell.trainable_weights if "recurrent_kernel" in w.name)
                if "kernel" in w.name:
                    W = w  # shape [input_dim, 4*units]
                if "bias" in w.name:
                    b = w

            if W is None or U is None or b is None:
                return

            units = gru_cell.units

            # Slice into gates: update (z), reset (r), candidate (h)
            W_z, W_r, W_h = tf.split(W, 3, axis=1)
            U_z, U_r, U_h = tf.split(U, 3, axis=1)

            # Bias in TF2 has shape [2*3*units], split and then sum input+recurrent
            b_input, b_recurrent = tf.unstack(b, axis=0)  # each is [3*units]

            # Split into gates
            b_z_i, b_r_i, b_h_i = tf.split(b_input, 3, axis=0)
            b_z_r, b_r_r, b_h_r = tf.split(b_recurrent, 3, axis=0)

            # Combine input + recurrent for each gate
            b_z = b_z_i + b_z_r
            b_r = b_r_i + b_r_r
            b_h = b_h_i + b_h_r

            # === Condition focuses on forget/update gate ===
            # ||U_r||_inf
            U_r_norm = tf.reduce_max(tf.reduce_sum(tf.abs(U_r), axis=1))

            # sigma(||[W_f, U_f, b_f]||_inf), here "forget" is the reset/update gate
            b_f_2d = tf.expand_dims(b_r, axis=1)  # pick r-gate (can adapt if using z instead)
            M_f = tf.concat([W_r, U_r, b_f_2d], axis=1)
            row_sums_f = tf.reduce_sum(tf.abs(M_f), axis=1)
            norm_inf_f = tf.reduce_max(row_sums_f)
            sigma_f = tf.sigmoid(norm_inf_f)

            lhs = U_r_norm * sigma_f

            if lhs >= 1.0:
                scale = 0.9 / lhs  # shrink factor

                # Split into gates
                U_z, U_r, U_h = tf.split(U, 3, axis=1)

                # Row sums for U_r
                row_sums = tf.reduce_sum(tf.abs(U_r), axis=1, keepdims=True)  # [units, 1]

                # Condition: row_sums * sigma_f < 1
                thresh = 1.0 / sigma_f
                mask = row_sums >= thresh   # rows that violate condition

                # Row-wise scaling: only shrink violating rows
                scaling_factors = tf.where(mask, scale * tf.ones_like(row_sums), tf.ones_like(row_sums))

                U_r_projected = U_r * scaling_factors

                # Recombine and assign
                U_new = tf.concat([U_z, U_r_projected, U_h], axis=1)
                U.assign(U_new)

                print(f"[ISS] GRU: projected {tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()} violating rows.")

        elif cell_type == "lstm":
            # Grab the LSTMCell
            lstm_cell = self._cell.KGCELL   # or QCELL if enforcing on Q-network

            # Extract weight matrices
            # In tf.keras, LSTMCell packs all gates [i, f, c, o] together in kernel and recurrent_kernel
            for w in lstm_cell.trainable_weights:
                if "recurrent_kernel" in w.name:
                    U = w  # shape [units, 4*units]
                if "kernel" in w.name:
                    W = w  # shape [input_dim, 4*units]
                if "bias" in w.name:
                    b = w

            units = lstm_cell.units

            # Split into gate matrices
            # Ordering in TF is: [i, f, c, o]
            U_i, U_f, U_c, U_o = tf.split(U, num_or_size_splits=4, axis=1)
            W_i, W_f, W_c, W_o = tf.split(W, num_or_size_splits=4, axis=1)
            b_i, b_f, b_c, b_o = tf.split(b, num_or_size_splits=4, axis=0)

            # Compute induced inf-norms
            def compute_bar_sigma(W, U, b):
                b_2d = tf.expand_dims(b, axis=1)  # [H, 1]
                M = tf.concat([W, U, b_2d], axis=1)  # [H, d_in+H+1]
                row_sums = tf.reduce_sum(tf.abs(M), axis=1)   # l₁ per row
                norm_inf = tf.reduce_max(row_sums)            # induced inf-norm
                return 1 / (1 + np.exp(-norm_inf))

            sigma = lambda x: 1 / (1 + np.exp(-x))  # logistic

            sigma_f = sigma(compute_bar_sigma(W_f, U_f, b_f))
            sigma_i = sigma(compute_bar_sigma(W_i, U_i, b_i))
            sigma_o = sigma(compute_bar_sigma(W_o, U_o, b_o))
            U_c_norm = tf.norm(U_c, ord=np.inf).numpy()

            cond1 = (1 + sigma_o) * sigma_f
            cond2 = (1 + sigma_o) * sigma_i * U_c_norm

            if cond1 >= 1.0 or cond2 >= 1.0:
                scale = 0.9 / max(cond1, cond2)  # shrink to stay within bounds
                new_U = U * scale
                U.assign(new_U)
                print(f"[ISS] Scaled LSTM recurrent weights by {scale:.3f} to satisfy ISS condition.")
    
    def training(self, model, Train_Obs, Train_Target, Valid_Obs, Valid_Target,
                 test_obs, test_targets, epochs, batch_size, 
                 x_epoch, record, fig, ax0, draw_fig):
        """
        training procedure
        depending on the task, appropriate loss function is taken account
        """

        #valid batching
        batch_size = batch_size
        Valid_Ybatch = []
        Valid_Ubatch = []
        Valid_ValidBatch = []
        for bid in range(int(len(Valid_Target)/batch_size)):
            Valid_Ybatch.append( Valid_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Valid_Obs)/batch_size)):
            Valid_Ubatch.append( Valid_Obs[bid*batch_size:(bid+1)*batch_size])

        Valid_Ybatch = np.array(Valid_Ybatch)
        Valid_Ubatch = np.array(Valid_Ubatch)

        
        #train batching
        batch_size = batch_size
        Ybatch = []
        Ubatch = []
        ValidBatch = []
        for bid in range(int(len(Train_Target)/batch_size)):
            Ybatch.append( Train_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Train_Obs)/batch_size)):
            Ubatch.append( Train_Obs[bid*batch_size:(bid+1)*batch_size])

        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)

        
        Training_Loss = []
        min_test_loss = 500
        min_train_loss = 500
        with open(self.result_path + '/trainloss.txt', 'w') as trl:
            with open(self.result_path + '/testloss.txt', 'w') as tel:
                
                for epoch in range(epochs):
                    train_loss_epoch =[]
                    for i in range(len(Ybatch)):
                        NetIn = Ubatch[i]
                        reinforce_loss = 0
                        with tf.GradientTape() as tape:
                            preds, logp_list = model(NetIn)
                            reinforce_loss = self.reinforce_loss(Ybatch[i], preds, logp_list)

                        with tf.GradientTape() as tape2:
                            preds, _ = model(NetIn)  # recompute
                            phi_loss = self.gaussian_nll(Ybatch[i], preds)
                        
                        print('epoch: %d  reinforce_loss: %s' % (epoch, reinforce_loss.numpy()))
                        if np.isnan(reinforce_loss.numpy()):
                            break
                        dynamic_variables = model._layer_rkn.cell._coefficient_net.weights
                        gradients = tape.gradient(reinforce_loss, dynamic_variables)
                        tf.keras.optimizers.Adam(learning_rate = self.lr, clipnorm=5.0).apply_gradients(zip(gradients, dynamic_variables))
                        
                        print('epoch: %d  base_loss: %s' % (epoch, phi_loss.numpy()))
                        if np.isnan(phi_loss.numpy()):
                            break

                        phi_vars =  [v for v in model.trainable_variables if all(v is not d for d in dynamic_variables)]
                        grads_phi = tape2.gradient(phi_loss, phi_vars)
                        tf.keras.optimizers.Adam(learning_rate = self.lr, clipnorm=5.0).apply_gradients(zip(grads_phi, phi_vars))

                        if i %10==0:
                            rand_sel = np.random.randint(0, len(Valid_Ubatch))
                            val_preds, val_logp_list = model(Valid_Ubatch[rand_sel])
                            val_reinforce_loss = self.reinforce_loss(Valid_Ybatch[rand_sel], val_preds, val_logp_list)
                            val_phi_loss = self.gaussian_nll(Valid_Ybatch[rand_sel], val_preds)
                            val_loss = val_reinforce_loss + val_phi_loss
                            print('val loss: %s' % (val_loss.numpy()))
                        
                        loss = phi_loss + reinforce_loss
                        print('epoch: %d  total_loss: %s' % (epoch, loss.numpy()))
                        if np.isnan(loss.numpy()):
                            break
                        train_loss_epoch.append(loss)
                        Training_Loss.append(loss)  
                        # === Enforce ISS stability constraint ===
                        if self.cell_type.lower() == 'gin':
                            model.enforce_iss_condition(cell_type=model._layer_rkn.cell.cell_type)

                    average_train_loss = np.mean(np.array(train_loss_epoch))

                    
                    
                    # if (average_train_loss < min_train_loss):
                    #     min_train_loss = average_train_loss
                    #     trl.write("*********************************min_train_loss:{} \n".format(min_train_loss))
                    # average_test_loss = np.mean(np.array(self.testing(model, test_obs, test_targets, batch_size)))
                    # if (average_test_loss < min_test_loss):
                    #     min_test_loss = average_test_loss
                    #     tel.write("*********************************min_train_loss:{} \n".format(min_test_loss))
                    # tel.write("epoch: {}, test_loss:{} \n".format(epoch, average_test_loss))
                    # trl.write("epoch: {}, train_loss:{} \n".format(epoch, average_train_loss))
                    # if np.isnan(average_train_loss):
                    #     break
                    # self.draw_curve(epoch, average_train_loss, average_test_loss, record, fig, ax0, x_epoch, batch_size, batch_size)

                    if ((epoch+1) % self.lr_decay_it == 0):
                        self.lr *= self.lr_decay
                        print(self.lr)
                    
            tel.close()
        trl.close()
        return Training_Loss
    
    def testing(self, model, test_obs, test_targets, batch_size):
        """
        testing procedure
        depending on the task, appropriate loss function is taken account
        """
        
        batch_size = batch_size
        Ybatch = []
        Ubatch = []
        ValidBatch = []
        for bid in range(int(len(test_targets)/batch_size)):
            Ybatch.append( test_targets[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(test_obs)/batch_size)):
            Ubatch.append( test_obs[bid*batch_size:(bid+1)*batch_size])

        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)

        
        Test_Loss = []
        for i in range(len(Ybatch)):
            NetIn = Ubatch[i]
            preds = model(NetIn)
            loss = self.gaussian_nll(Ybatch[i], preds)
            print('test loss: %s' % (loss.numpy()))
            Test_Loss.append(loss)
        return Test_Loss

    def draw_curve(self, epoch, train_loss, test_loss, record, fig, ax0, x_epoch, batch_size_train, batch_size_test):
        # global record
        record['train_loss'].append(train_loss)
        record['test_loss'].append(test_loss)


        x_epoch.append(epoch)
        ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
        ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
        
        if epoch == 0:
            ax0.legend()
        fig.savefig(self.result_path +"/train_{:.3f}_{}.jpg".format(batch_size_train, batch_size_test))
    # helpers
    @staticmethod
    def _prop_through_layers(inputs, layers):
        """propagation"""
        h = inputs
        for layer in layers:
            h = layer(h)
        return h

    @staticmethod
    def _time_distribute_layers(layers):
        """wraps layers with k.layers.TimeDistributed"""
        td_layers = []
        for l in layers:
            td_layers.append(k.layers.TimeDistributed(l))
        return td_layers
    
