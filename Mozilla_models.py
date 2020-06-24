from keras.models import Model
from keras.layers import CuDNNGRU, Input, Dense, BatchNormalization, Activation, TimeDistributed
import keras.backend as K
import numpy as np
from keras_attention.models.custom_recurrents import AttentionDecoder
from keras.layers.wrappers import Bidirectional
from keras import regularizers


def repeat_encoding(input_tensors):
    """
    https://github.com/keras-team/keras/issues/7949#issuecomment-555743984
    repeats encoding based on input shape of the model
    :param input_tensors: [  tensor : encoding ,  tensor : sequential_input]
    :return: tensor : encoding repeated input-shape[1]times
    """
    sequential_input = input_tensors[1]
    to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
    # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
    one_matrix = K.ones_like(sequential_input[:,:,:1])
    
    
    # do a mat mul
    return K.batch_dot(one_matrix,to_be_repeated)
    

def import_models(ind, n_lstm, n_seed, n_mel, seq_length):
    np.random.seed(n_seed)
    K.tf.set_random_seed(n_seed)

    if ind == 'GRU_autoencoder':
        x_in_encoder = Input(shape = (None, n_mel), name='in_signal');
        x_in_pre = Input(shape = (None, n_mel), name='in_pre');       
        
        
        x_enc_lstm, state_h = CuDNNGRU(n_lstm, return_sequences=False, return_state = True ,
                      name='encoder')(x_in_encoder)
        
        state_h = Dense(n_lstm)(state_h)
        state_h = BatchNormalization()(state_h)
        state_h = Activation('tanh')(state_h)
        # Onset net
        x = CuDNNGRU(n_lstm, return_sequences=True)(x_in_pre)
        
        # Autoencoder
        x = CuDNNGRU(n_lstm,return_sequences=True)(x,initial_state=[state_h])       
        x = TimeDistributed(Dense(n_mel, activation = 'tanh'), name='out_main')(x)  
        
    
        model = Model(inputs = [x_in_encoder,x_in_pre],
                      outputs=[x])
    if ind == 'GRU_attention':
        # https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
        x_in_encoder = Input(shape = (seq_length, n_mel), name='in_signal');             
        x_enc = Bidirectional(CuDNNGRU(n_lstm, return_sequences=True, 
                                       #kernel_regularizer= regularizers.l2(0.01),
                                       name='encoder'))(x_in_encoder)
        x_enc = BatchNormalization()(x_enc)
        x = AttentionDecoder(n_lstm, n_mel,name='out_main')(x_enc)
#         x = TimeDistributed(Dense(n_mel, activation = 'tanh'), name='out_main')(x)  
        
        model = Model(inputs = x_in_encoder, outputs = x)

    return model
