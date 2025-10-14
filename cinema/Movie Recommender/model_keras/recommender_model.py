import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="MyLayers")
class NormLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,x):
        return tf.linalg.l2_normalize(x, axis=1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct the layer from the config dictionary
        return cls(**config)
    
def get_model(user_train_shape, movies_train_shape):
    num_user_features = user_train_shape - 2
    num_movies_features = movies_train_shape - 3

    num_outputs = 32
    tf.random.set_seed(1)
    user_NN = tf.keras.models.Sequential([     
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs)
    ])

    movies_NN = tf.keras.models.Sequential([    
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs) 
    ])

    # create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    vu = user_NN(input_user)
    vu = NormLayer()(vu)

    # create the item input and point to the base network
    input_movie = tf.keras.layers.Input(shape=(num_movies_features,))
    vm = movies_NN(input_movie)
    vm = NormLayer()(vm)

    # compute the dot product of the two vectors vu and vm
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    # specify the inputs and output of the model
    model = tf.keras.Model([input_user, input_movie], output)

    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,loss=cost_fn)

    return model

def train_model(model, tf_dataset, epochs=1):

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
    )

    model.fit(tf_dataset, epochs=epochs, callbacks=[early_stop])
    
    return model