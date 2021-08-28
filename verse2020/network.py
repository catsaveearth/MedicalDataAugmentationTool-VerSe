import tensorflow as tf
from tensorflow.keras.layers import Conv3D, AveragePooling3D, AlphaDropout, Activation, Dropout 
# tf.keras.layers로부터 3d convolution, 3d average pooling, alpha dropout, activation, dropout layer를 불러옴

from tensorflow.keras.regularizers import l2
# tf.keras.regularizers로부터 l2 norm 정규화 모듈(?)을 불러옴

from tensorflow_train_v2.layers.initializers import he_initializer, selu_initializer
# tensorflow_train_v2? 검색해도 안나오는데 그냥 TF2라고 생각하면 될듯? 거기서 initializer 두개를 불러옴

from tensorflow_train_v2.layers.layers import Sequential, ConcatChannels, UpSampling3DLinear, UpSampling3DCubic
# tf2에서 sequential, concatchannels, upsampling 3d linear, upsampling 3d cubic을 불러옴 => u-net에 필요한 친구들

from tensorflow_train_v2.networks.unet_base import UnetBase
# tf2에서 UnetBaes를 불러옴.

class UnetAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    u-net class인데 average pooling과 linear upsampling으로 구성된 네트워크 클래스.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):

        #클래스 인자 하나씩 매칭시켜주기
        super(UnetAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level): #u-net의 왼쪽부분 구성(크기 감소)
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling3D([2] * 3, data_format=self.data_format)

    def upsample(self, current_level): #u-net의 오른쪽 부분 구성 (크기 증가)
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling3DLinear([2] * 3, data_format=self.data_format)

    def combine(self, current_level): #u-net의 사이 (skip-connection) 를 구현 (왼쪽의 기존 크기의 얘 + 오른쪽의 upsampling된 얘)
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)
        # return Add()

    #아래 두 block 코드는 마지막 sequential의 name부분을 제외하면 코드가 똑같음
    def contracting_block(self, current_level):
        """
        #왼쪽의 contracting path를 구성하는 block들
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats): #convolution을 몇번 반복하는지
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout: #drop out 종류에 따라 맞는 얘를 넣어주기
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level)) #그 level에 맞는 레이어 시퀸스가 뚝딱!

    def expanding_block(self, current_level):
        """
        #오른쪽의 expanding path를 구성하는 block들
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='expanding' + str(current_level))

    def conv(self, current_level, postfix):
        """
        3D convolution 코드!
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv3D(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)


def activation_fn_output_kernel_initializer(activation):
    """
    Return actual activation function and kernel initializer.
    활성화 함수와 커널 initializer를 리턴함.
    :param activation: Activation function string.
    :return: activation_fn, kernel_initializer

    6가지 종류를 제공
    """
    if activation == 'none':
        activation_fn = None
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.001)
    elif activation == 'tanh':
        activation_fn = tf.tanh
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.001)
    elif activation == 'abs_tanh':
        activation_fn = lambda x, *args, **kwargs: tf.abs(tf.tanh(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.0001)
    elif activation == 'square_tanh':
        activation_fn = lambda x: tf.tanh(x * x)
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'inv_gauss':
        activation_fn = lambda x: 1.0 - tf.math.exp(-tf.square(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'squash':
        a = 5
        b = 1
        l = 1
        activation_fn = lambda x: 1.0 / (l * b) * tf.math.log((1.0 + tf.math.exp(b * (x - (a - l / 2.0)))) / (1.0 + tf.math.exp(b * (x - (a + l / 2.0)))))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'sigmoid':
        activation_fn = lambda x: tf.nn.sigmoid(x - 5)
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    return activation_fn, kernel_initializer


#vertebra localization에 쓰이는 친구
class SpatialConfigurationNet(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self,
                 num_labels,
                 num_filters_base=64,
                 num_levels=4,
                 activation='relu',
                 data_format='channels_first',
                 padding='same',
                 local_activation='none',
                 spatial_activation='none',
                 spatial_downsample=8,
                 dropout_ratio=0.0):
        """
        Initializer.
        :param num_labels: Number of outputs.
        :param num_filters_base: Number of filters for the local appearance and spatial configuration sub-networks.
        :param num_levels: Number of levels for the local appearance and spatial configuration sub-networks.
        :param activation: Activation of the convolution layers ('relu', 'lrelu', or 'selu').
        :param data_format: 'channels_first' or 'channels_last'
        :param padding: Convolution padding.
        :param local_activation: Activation function of local appearance output.
        :param spatial_activation: Activation function of spatial configuration output.
        :param spatial_downsample: Downsample factor for spatial configuration output.
        :param dropout_ratio: The dropout ratio after each convolution layer.
        """
        super(SpatialConfigurationNet, self).__init__()
        self.unet = UnetAvgLinear3D
        self.data_format = data_format
        self.num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'selu':
            activation_fn = tf.nn.selu
            kernel_initializer = selu_initializer
            alpha_dropout = True
        local_activation_fn, local_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(local_activation)
        spatial_activation_fn, spatial_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(spatial_activation)
        self.downsampling_factor = spatial_downsample
        self.scnet_local = self.unet(num_filters_base=self.num_filters_base, num_levels=num_levels, kernel_initializer=kernel_initializer, alpha_dropout=alpha_dropout, activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.local_heatmaps = Sequential([Conv3D(num_labels, [1] * 3, name='local_heatmaps', kernel_initializer=local_heatmap_layer_kernel_initializer, activation=None, data_format=data_format, padding=padding),
                                          Activation(local_activation_fn, dtype='float32', name='local_heatmaps')])
        self.downsampling = AveragePooling3D([self.downsampling_factor] * 3, name='local_downsampling', data_format=data_format)
        self.scnet_spatial = self.unet(num_filters_base=self.num_filters_base, num_levels=num_levels, repeats=1, kernel_initializer=kernel_initializer, activation=activation_fn, alpha_dropout=alpha_dropout, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.spatial_heatmaps = Conv3D(num_labels, [1] * 3, name='spatial_heatmaps', kernel_initializer=spatial_heatmap_layer_kernel_initializer, activation=None, data_format=data_format, padding=padding)
        self.upsampling = Sequential([UpSampling3DCubic([self.downsampling_factor] * 3, name='spatial_upsampling', data_format=data_format),
                                      Activation(spatial_activation_fn, dtype='float32', name='spatial_heatmaps')])

    def call(self, inputs, training, **kwargs):
        """
        Call model.
        :param inputs: Input tensors.
        :param training: If True, use training mode, otherwise testing mode.
        :param kwargs: Not used.
        :return: (heatmaps, local_heatmaps, spatial_heatmaps) tuple.
        """
        node = self.scnet_local(inputs, training=training)
        local_heatmaps = node = self.local_heatmaps(node, training=training)
        node = self.downsampling(node, training=training)
        node = self.scnet_spatial(node, training=training)
        node = self.spatial_heatmaps(node, training=training)
        spatial_heatmaps = self.upsampling(node, training=training)
        heatmaps = local_heatmaps * spatial_heatmaps

        return heatmaps, local_heatmaps, spatial_heatmaps


class Unet(tf.keras.Model):
    """
    The Unet.
    그냥 U-net
    """
    def __init__(self,
                 num_labels,
                 num_filters_base=64,
                 num_levels=4,
                 activation='relu',
                 data_format='channels_first',
                 padding='same',
                 dropout_ratio=0.0,
                 heatmap_initialization=False,
                 **kwargs):
        """
        Initializer.
        :param num_labels: Number of outputs. 아웃풋 라벨의 수
        :param num_filters_base: Number of filters for the local appearance and spatial configuration sub-networks.
        :param num_levels: Number of levels for the local appearance and spatial configuration sub-networks. u-net 단계 (level)의 수
        :param activation: Activation of the convolution layers ('relu', 'lrelu', or 'selu'). 활성화함수 종류
        :param data_format: 'channels_first' or 'channels_last' 데이터포맷 NCHW인지 CNHW인지 그런건가?
        :param padding: Convolution padding.
        :param dropout_ratio: The dropout ratio after each convolution layer. #dropout 비율
        :param **kwargs: Not used.
        """
        super(Unet, self).__init__()
        self.data_format = data_format
        num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        if heatmap_initialization:
            heatmap_layer_kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.0001)
        else:
            heatmap_layer_kernel_initializer = he_initializer

        #u-net을 UnetAvgLinear3D로 지정해주기
        self.unet = UnetAvgLinear3D(num_filters_base=num_filters_base, num_levels=num_levels, kernel_initializer=he_initializer, activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        
        #예측하는 부분? 3D Conv = activation으로 구성됨.
        self.prediction = Sequential([Conv3D(num_labels, [1] * 3, name='prediction', kernel_initializer=heatmap_layer_kernel_initializer, activation=None, data_format=data_format, padding=padding),
                                      Activation(None, dtype='float32', name='prediction')])
        self.single_output = num_labels == 1 #output이 하나라면 True가 된다.

    def call(self, inputs, training, **kwargs):
        """
        Call model.
        :param inputs: Input tensors.
        :param training: If True, use training mode, otherwise testing mode. #Train인지 test인지 여부 (boolean)
        :param kwargs: Not used.
        :return: prediction
        """
        node = self.unet(inputs, training=training)
        prediction = self.prediction(node, training=training)

        if self.single_output: #이부분은 왜지? 
            return prediction
        else:
            return prediction, prediction, prediction
