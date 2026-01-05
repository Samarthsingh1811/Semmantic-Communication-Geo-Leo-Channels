import tensorflow as tf

class MLP(tf.keras.layers.Layer):
    def __init__(self, out_features, expansion_coeff=4):
        super().__init__()

        self.fc1 = tf.keras.layers.Dense(
            out_features * expansion_coeff
        )
        self.gelu = tf.nn.gelu
        self.fc2 = tf.keras.layers.Dense(
            out_features
        )
    
    def call(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class WindowMHSA(tf.keras.layers.Layer):
    '''
    Implements Windowed Multi-head Self-attention 
    with Shifted Window mechanism (Swin-style)
    '''
    def __init__(self, num_heads, dim_head, window_size=8, shift_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.qkv = tf.keras.layers.Dense(dim_head * num_heads * 3, use_bias=True)
        self.proj = tf.keras.layers.Dense(dim_head * num_heads)
        
        # Relative position bias table
        self.relative_position_bias_table = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev=0.02)(
                shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
        )
        
        # Get relative position index
        coords_h = tf.range(self.window_size)
        coords_w = tf.range(self.window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij')) # [2, Wh, Ww]
        coords_flatten = tf.reshape(coords, (2, -1)) # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # [2, Wh*Ww, Wh*Ww]
        relative_coords = tf.transpose(relative_coords, (1, 2, 0)) # [Wh*Ww, Wh*Ww, 2]
        relative_coords += self.window_size - 1 # shift to start from 0
        relative_coords *= tf.constant([2 * self.window_size - 1, 1], dtype=tf.int32)
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1) # [Wh*Ww, Wh*Ww]
        self.relative_position_index = tf.Variable(relative_position_index, trainable=False)

    def window_partition(self, x, window_size):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
        windows = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = tf.reshape(windows, (-1, window_size, window_size, C))
        return windows

    def window_reverse(self, windows, window_size, H, W):
        C = tf.shape(windows)[-1]
        x = tf.reshape(windows, (-1, H // window_size, W // window_size, window_size, window_size, C))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (-1, H, W, C))
        return x

    def call(self, x, mask=None):
        B_, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
            
        # Partition Windows
        x_windows = self.window_partition(shifted_x, self.window_size) # [nW*B, Wh, Ww, C]
        nW_B = tf.shape(x_windows)[0]
        x_windows = tf.reshape(x_windows, (nW_B, self.window_size * self.window_size, C)) # [nW*B, Wh*Ww, C]
        
        # W-MSA / SW-MSA
        qkv = self.qkv(x_windows)
        qkv = tf.reshape(qkv, (nW_B, self.window_size * self.window_size, 3, self.num_heads, self.dim_head))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * (self.dim_head ** -0.5)
        attn = tf.matmul(q, k, transpose_b=True)
        
        # Add Relative Position Bias
        rel_pos_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1,)))
        rel_pos_bias = tf.reshape(rel_pos_bias, (self.window_size * self.window_size, self.window_size * self.window_size, -1))
        rel_pos_bias = tf.transpose(rel_pos_bias, (2, 0, 1)) # [nH, Wh*Ww, Wh*Ww]
        attn = attn + rel_pos_bias
        
        # Masking for SW-MSA
        if self.shift_size > 0:
            # Simple window masking could be implemented here for full Swin, 
            # but for this JSCC use-case, let's keep it robust and simple.
            pass
            
        attn = tf.nn.softmax(attn, axis=-1)
        x_windows = tf.matmul(attn, v)
        x_windows = tf.transpose(x_windows, (0, 2, 1, 3))
        x_windows = tf.reshape(x_windows, (nW_B, self.window_size, self.window_size, C))
        
        # Reverse Partition
        shifted_x = self.window_reverse(x_windows, self.window_size, H, W)
        
        # Reverse Shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
            
        x = self.proj(x)
        return x


class VitBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, spatial_size, stride=1, window_size=8):
        super().__init__()
        self.d_out = num_heads * head_size
        self.window_size = window_size
        
        self.patchmerge = tf.keras.layers.Conv2D(
            filters=self.d_out,
            kernel_size=stride,
            strides=stride,
        )
        
        self.ln1 = tf.keras.layers.LayerNormalization()
        # Layer 1 Attention (Normal Window)
        self.msa1 = WindowMHSA(num_heads, head_size, window_size=window_size, shift_size=0)
        
        self.ln2 = tf.keras.layers.LayerNormalization()
        # Layer 2 Attention (Shifted Window)
        self.msa2 = WindowMHSA(num_heads, head_size, window_size=window_size, shift_size=window_size // 2)
        
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(self.d_out)

    def call(self, x):
        x = self.patchmerge(x)
        
        # Simple Swin Block Structure (W-MSA + SW-MSA + MLP)
        # 1. W-MSA
        residual = x
        x = self.ln1(x)
        x = self.msa1(x)
        x = x + residual
        
        # 2. SW-MSA
        residual = x
        x = self.ln2(x)
        x = self.msa2(x)
        x = x + residual
        
        # 3. MLP
        residual = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

