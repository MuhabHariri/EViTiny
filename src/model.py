import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- Layers & blocks (exact behavior preserved) ----------

class ECALayer(layers.Layer):
    def __init__(self, channels, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        t = abs((np.log(channels) / np.log(2.0) + b) / gamma)
        k = int(t) if int(t) % 2 else int(t) + 1
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.conv = layers.Conv1D(1, kernel_size=k, padding="same", activation='sigmoid', use_bias=False)

    def call(self, inputs):
        y = self.avg_pool(inputs)
        y = tf.expand_dims(y, -1)
        y = self.conv(y)
        y = tf.squeeze(y, -1)
        y = tf.reshape(y, [-1, 1, 1, self.channels])
        return inputs * y

def create_autoencoder(input_shape=(256, 256, 3)):
    input_img = layers.Input(shape=input_shape)

    x = layers.SeparableConv2D(34, (3, 3), padding='same', strides=(2, 2))(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = ECALayer(34)(x)

    x = layers.SeparableConv2D(52, (3, 3), padding='same', strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = ECALayer(52)(x)

    x = layers.SeparableConv2D(66, (3, 3), padding='same', strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    encoded = x
    x = layers.SpatialDropout2D(0.15)(encoded)
    x = ECALayer(66)(x)

    x = layers.Conv2DTranspose(66, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = ECALayer(66)(x)

    x = layers.Conv2DTranspose(52, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = ECALayer(52)(x)

    x = layers.Conv2DTranspose(34, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.SpatialDropout2D(0.15)(x)
    x = ECALayer(34)(x)

    x = layers.Conv2D(3, (3, 3), padding='same')(x)
    decoded = layers.LeakyReLU(alpha=0.3)(x)

    return models.Model(input_img, decoded)

def positional_encoding(shape, embedding_size):
    height, width, _ = shape
    position_enc = np.zeros((height * width, embedding_size), dtype=np.float32)
    positions = np.arange(height * width)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_size, 2) * -(np.log(10000.0) / embedding_size))
    position_enc[:, 0::2] = np.sin(positions * div_term)
    position_enc[:, 1::2] = np.cos(positions * div_term)
    position_enc = position_enc.reshape(height, width, embedding_size)
    return position_enc

class RGCCG_module(tf.keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.global_context_gate = tf.keras.layers.Dense(embedding_size, activation="sigmoid")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.context_weight = self.add_weight(
            name='context_weight', shape=(input_shape[-1], self.embedding_size),
            initializer='glorot_uniform', trainable=True
        )
        self.bias = self.add_weight(
            name='bias', shape=(self.embedding_size,), initializer='zeros', trainable=True
        )

    def call(self, inputs):
        global_context = tf.reduce_mean(inputs, axis=1, keepdims=True) @ self.context_weight
        global_context = self.layer_norm(global_context)
        global_context = global_context + self.bias
        gated_context = self.global_context_gate(global_context)
        return inputs + gated_context * inputs

class NonLinear_Projection(tf.keras.layers.Layer):
    def __init__(self, projection_dim, activation=tf.nn.gelu, initializer='he_normal', **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.activation = activation
        self.initializer = initializer

    def build(self, input_shape):
        self.projection = tf.keras.layers.Dense(
            self.projection_dim, use_bias=False, activation=self.activation,
            kernel_initializer=self.initializer
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        return self.norm(self.projection(x))

class NonLinear_and_MHSA(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.query_projection = NonLinear_Projection(self.embedding_size, initializer='glorot_uniform')
        self.key_projection = NonLinear_Projection(self.embedding_size, initializer='glorot_uniform')
        self.value_projection = NonLinear_Projection(self.embedding_size, initializer='glorot_uniform')
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_size // num_heads, dropout=0.25
        )

    def call(self, x, mask=None):
        query = self.query_projection(self.query_projection(x))
        key = self.key_projection(self.key_projection(x))
        value = self.value_projection(self.value_projection(x))
        attention_output = self.multi_head_attention(query, key, value, attention_mask=mask)
        return attention_output

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        binary_tensor = tf.floor(random_tensor)
        return tf.math.divide(x, keep_prob) * binary_tensor

class Core_ViTiny(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_size, drop_path_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention = NonLinear_and_MHSA(num_heads, embedding_size)
        self.global_context = RGCCG_module(embedding_size)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention_scale = self.add_weight(name='attention_scale', shape=(1,), initializer='ones', trainable=True)
        self.mlp_scale = self.add_weight(name='mlp_scale', shape=(1,), initializer='ones', trainable=True)
        self.drop_path1 = StochasticDepth(drop_prob=drop_path_rate)
        self.drop_path2 = StochasticDepth(drop_prob=drop_path_rate)

    def call(self, x, training=None):
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x)
        attn_output = attn_output * self.attention_scale
        attn_output = self.drop_path1(attn_output, training=training)
        x = x + attn_output

        x = self.global_context(x)
        norm_x = self.norm2(x)
        output = norm_x
        output = output * self.mlp_scale
        output = self.drop_path2(output, training=training)
        return x + output

def mlp_head(x, hidden_units, output_units):
    for units in hidden_units:
        x = layers.Dense(units)(x)
    return layers.Dense(output_units, activation='softmax')(x)

class WeightedResidualConnection(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, inputs, residual):
        weighted_residual = residual * self.weight
        combined = layers.add([inputs, weighted_residual])
        return self.layer_norm(combined)

def deep_resnet_unit(
    x,
    num_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    dropout_rate=0.05,
    groups=None,  # use 2 in the middle loop, None elsewhere (matches original)
):
    x = layers.DepthwiseConv2D(kernel_size, padding='same', strides=strides)(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    if groups is None:
        x = layers.Conv2D(num_filters, (1, 1), padding='same')(x)
    else:
        x = layers.Conv2D(num_filters, (1, 1), padding='same', groups=groups)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = ECALayer(num_filters)(x)
    return x


def deep_resnet_block(
    x,
    num_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    additional_layers=0,
    dropout_rate=0.05,
    groups=2,
):
    residual = x

    # first unit (with possible stride), no groups
    x = deep_resnet_unit(
        x, num_filters, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate, groups=None
    )

    # middle units (additional_layers), groups=2 as in original
    for _ in range(additional_layers):
        x = deep_resnet_unit(
            x, num_filters, kernel_size=kernel_size, strides=(1, 1), dropout_rate=dropout_rate, groups=2
        )

    # last unit, no groups
    x = deep_resnet_unit(
        x, num_filters, kernel_size=kernel_size, strides=(1, 1), dropout_rate=dropout_rate, groups=None
    )

    if residual.shape[-1] != num_filters or strides != (1, 1):
        residual = layers.Conv2D(num_filters, (1, 1), strides=strides, padding='same')(residual)
        residual = layers.BatchNormalization()(residual)

    weighted_skip = WeightedResidualConnection()
    x = weighted_skip(x, residual)
    x = tf.keras.activations.gelu(x)
    return x


def ViTiny_First_block(encoder_output, num_heads, num_transformer_layers, embedding_size=66):
    _, height, width, channels = encoder_output.shape
    num_patches = height * width

    position_enc = positional_encoding((height, width, channels), embedding_size)
    position_enc = tf.expand_dims(position_enc, axis=0)
    dense_output = layers.Dense(embedding_size)(encoder_output)
    dense_output = layers.LayerNormalization(epsilon=1e-6)(dense_output)
    x = layers.Add()([dense_output, position_enc])
    x = layers.Flatten()(x)
    x = layers.Reshape((num_patches, embedding_size))(x)
    for _ in range(num_transformer_layers):
        x = Core_ViTiny(num_heads, embedding_size)(x)
    transformer_output = layers.Reshape((height, width, embedding_size))(x)
    transformer_output = WeightedResidualConnection()(dense_output, transformer_output)
    transformer_output = tf.reshape(transformer_output, (-1, height, width, embedding_size))
    return transformer_output

def ViTiny_Second_block(encoder_output, num_heads, num_transformer_layers, embedding_size=66):
    _, height, width, channels = encoder_output.shape
    num_patches = height * width

    position_enc = positional_encoding((height, width, channels), embedding_size)
    position_enc = tf.expand_dims(position_enc, axis=0)
    dense_output = layers.Dense(embedding_size)(encoder_output)
    dense_output = layers.LayerNormalization(epsilon=1e-6)(dense_output)
    x = layers.Add()([dense_output, position_enc])
    x = layers.Flatten()(x)
    x = layers.Reshape((num_patches, embedding_size))(x)
    for _ in range(num_transformer_layers):
        x = Core_ViTiny(num_heads, embedding_size)(x)
    transformer_output = layers.Reshape((height, width, embedding_size))(x)
    transformer_output = WeightedResidualConnection()(dense_output, transformer_output)
    return transformer_output

def EViTiny(input2, embedding_size, num_heads_list, num_classes, num_transformer_layers):
    residual_1 = input2
    x = ViTiny_First_block(input2, num_heads_list, num_transformer_layers=1, embedding_size=66)
    x = deep_resnet_block(x, num_filters=132, kernel_size=(3, 3), strides=(1, 1), additional_layers=2, dropout_rate=0.015)

    residual_1 = layers.SpatialDropout2D(0.02)(residual_1)
    if input2.shape[-1] != x.shape[-1]:
        residual_1 = layers.Conv2D(filters=132, kernel_size=(1, 1), strides=(1, 1))(residual_1)
    weighted_skip = WeightedResidualConnection()
    x = weighted_skip(x, residual_1)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    residual_2 = x

    x = deep_resnet_block(x, num_filters=192, kernel_size=(3, 3), strides=(1, 1), additional_layers=2, dropout_rate=0.015)
    residual_2 = layers.SpatialDropout2D(0.02)(residual_2)
    if input2.shape[-1] != x.shape[-1]:
        residual_2 = layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1))(residual_2)
    weighted_skip = WeightedResidualConnection()
    x = weighted_skip(x, residual_2)

    combined_feature_map = ViTiny_Second_block(x, num_heads_list, num_transformer_layers=2, embedding_size=192)
    combined_feature_map = layers.SpatialDropout2D(0.025)(combined_feature_map)

    pooled_output = layers.AveragePooling2D(pool_size=(10, 10), strides=(6, 6))(combined_feature_map)
    pooled_output = layers.Dropout(0.5)(pooled_output)
    pooled_output = layers.BatchNormalization()(pooled_output)
    pooled_output = layers.Flatten()(pooled_output)

    hidden_units = []  # preserved
    output = mlp_head(pooled_output, hidden_units, num_classes)
    return output

def remove_spatial_dropout(encoder):
    inputs = tf.keras.Input(shape=encoder.input.shape[1:])
    x = inputs
    for layer in encoder.layers:
        if not isinstance(layer, layers.SpatialDropout2D):
            x = layer(x)
    return models.Model(inputs, x)
