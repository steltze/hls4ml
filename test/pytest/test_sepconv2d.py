from pathlib import Path
import tensorflow as tf
import numpy as np
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D

import hls4ml
import os
test_root_path = Path(__file__).parent

os.environ['XILINX_VITIS'] = "/opt/Xilinx/Vitis_HLS/2023.2/"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
os.environ['XILINX_VIVADO'] = "/opt/Xilinx/Vivado/2020.1/"
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

keras_conv2d = [SeparableConv2D]
padds_options = ['same']
chans_options = ['channels_last']
io_type_options = ['io_stream']
strides_options = [(1, 1)]
kernel_options = [ (4, 4)]
bias_options = [False]
reuse_factor = [17, 37, 64, 159, 500, 1024]
strategies = ['resource']

@pytest.mark.parametrize('conv2d', keras_conv2d)
@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('strides', strides_options)
@pytest.mark.parametrize('kernels', kernel_options)
@pytest.mark.parametrize('bias', bias_options)
@pytest.mark.parametrize(
    'backend, io_type',
    [
        ('Vivado', 'io_parallel'),
        ('Vitis', 'io_parallel'),
        ('Vivado', 'io_stream'),
        ('Vitis', 'io_stream'),
        ('Catapult', 'io_stream'),
    ],
)
def test_sepconv2d(conv2d, chans, padds, strides, kernels, bias, io_type, backend):
    model = tf.keras.models.Sequential()
    input_shape = (16, 16, 3)
    model.add(
        conv2d(
            filters=8,
            kernel_size=kernels,
            strides=strides,
            padding=padds,
            input_shape=input_shape,
            kernel_initializer='normal',
            use_bias=bias,
            data_format=chans,
        )
    )

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,16>',)
    stride_cfg = str(strides).replace(', ', '_').replace('(', '').replace(')', '')
    kernel_cfg = str(kernels).replace(', ', '_').replace('(', '').replace(')', '')
    output_dir = str(
        test_root_path
        / 'hls4mlprj_{}_{}_strides_{}_kernels_{}_{}_padding_{}_{}_rf_{}_strategy_{}'.format(
            conv2d.__name__.lower(), chans, stride_cfg, kernel_cfg, padds, backend, io_type, rf, strategy
        )
    )
    
    config['Model']['Strategy'] = strategy
    config['Model']['ReuseFactor'] = rf
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, io_type=io_type, backend=backend
    )
    
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)