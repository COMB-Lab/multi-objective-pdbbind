from . import layers_update_mobley as layers
import importlib
import keras.backend as K
import numpy as np
import copy
import tensorflow as tf
# import tensorflow_addons as tfa
from models.dcFeaturizer import atom_features as get_atom_features
importlib.reload(layers)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers, constraints, callbacks
import sys
from tensorflow.keras.callbacks import EarlyStopping
import rdkit
import pickle
import gc

class PGGCNModel(tf.keras.Model):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, l2=1e-3, dropout_rate=0.4, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense5 = tf.keras.layers.Dense(16, activation='relu', name='dense2', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense6 = tf.keras.layers.Dense(1, name='dense6', kernel_regularizer=regularizers.l2(l2), bias_regularizer=regularizers.l2(l2), kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                             kernel_initializer=tf.keras.initializers.Constant([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                             bias_initializer=tf.keras.initializers.Zeros(),
                                             kernel_regularizer=regularizers.l2(l2), 
                                             bias_regularizer=regularizers.l2(l2), 
                                             kernel_constraint=constraints.MaxNorm(maxnorm))

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
        
    def set_input_shapes(self, i_s):
        self.i_s = i_s

    def call(self, inputs, training=False):
        print("Inside call")
        physics_info = inputs[:, 0, 38:] 
        x_a = []
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38])
        x = self.ruleGraphConvLayer(x_a)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense5(x)
        x = self.dropout2(x, training=training)
        model_var = self.dense6(x)
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
        return tf.concat([out, physics_info], axis=1)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + K.abs(1 / K.mean(.2 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def physical_consistency_loss(y_true,y_pred,physics_info):
    dG_pred = y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    # Physical Inconsistency loss
    # Extract the components from physics_info
    host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)  
    guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1) 
    complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1)  
    # Calculate ΔG based on physics: ΔG = ΔGcomplex - (ΔGhost + ΔGguest)
    dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (tf.reduce_sum(host, axis=1, keepdims=True) + tf.reduce_sum(guest, axis=1, keepdims=True))
    phy_loss = K.sqrt(K.mean(K.square(dG_pred - dG_physics)))

    return phy_loss



def combined_loss(physics_hyperparam=0.0003):
    def loss_function(y_true, y_pred):
        # Extract prediction and physics info
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
        
        # Calculate individual loss components
        empirical_loss = pure_rmse(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        # Combine losses with weights
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        
        return total_loss
    
    return loss_function



def get_trained_model(X, y, epochs = 1, physics_weight = 0.00005,max_num_atoms = 2000, n_features = 53):
    m = PGGCNModel()
    m.addRule("sum", 0, 32)
    m.addRule("multiply", 32, 33)
    m.addRule("distance", 33, 36)
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    m.compile(loss=combined_loss(physics_weight), optimizer=opt)
    
    X_train = X
    input_shapes = []
    for i in range(len(X_train)):
        input_shapes.append(np.array(X_train[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X_train)):
        if X_train[i].shape[0] < max_num_atoms:
            new_list = np.zeros([max_num_atoms - X_train[i].shape[0], n_features])
            X_train[i] = np.concatenate([X_train[i], new_list], 0)
    
    X_train = np.array(X_train)
    x_c = copy.deepcopy(X_train)
    y_train = np.array(y)
    
       # Add memory cleanup after each epoch
    class MemoryCleanup(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            if hasattr(tf.keras.backend, 'clear_session'):
                tf.keras.backend.clear_session()
                
    hist = m.fit(X_train, y_train, epochs=epochs, batch_size=len(X_train),callbacks=[MemoryCleanup()])
    return hist.history['loss'], m, x_c

def test_model(X_test, y_test, m: PGGCNModel, max_num_atoms = 2000, n_features = 53):
    input_shapes = []
    for i in range(len(X_test)):
        input_shapes.append(np.array(X_test[i]).shape[0])
    m.set_input_shapes(input_shapes)
    for i in range(len(X_test)):
        if X_test[i].shape[0] < max_num_atoms:
            new_list = np.zeros([max_num_atoms - X_test[i].shape[0], n_features])
            X_test[i] = np.concatenate([X_test[i], new_list], 0)
    X_test = np.array(X_test)
    x_c = copy.deepcopy(X_test)
    y_test = np.array(y_test)
    y_pred_test = m.predict(X_test)
    y_pred_test = np.array(y_pred_test[:,0])
    y_difference = np.mean(np.abs(np.abs(y_test) - np.abs(y_pred_test)))
    eval = m.evaluate(X_test, y_test,batch_size=len(X_test))
    return eval,y_difference, x_c

def featurize(molecule, info):
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [0]
        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features)
def data_generator(PDBs, info):
    X = []
    for i, pdb in enumerate(PDBs):
        X.append(featurize(pdb, info[i]))
    return X
