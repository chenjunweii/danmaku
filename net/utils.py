from mxnet import nd, gluon

def conv2rnn(inputs, fixed_time = False, fixed_feature = False):
    b, c, t, f = inputs.shape
    if fixed_time:
        return inputs.swapaxes(1,2).swapaxes(0,1).reshape([t, b, -1])
    elif fixed_feature:
        return inputs.swapaxes(1,2).swapaxes(0,1).reshape([t, b, ])
    elif c == 1:
        return inputs.reshape([b, t, f]).swapaxes(0,1).reshape([t, b, f])
def rnn2conv(inputs):
    t, b, f = inputs.shape
    return inputs.swapaxes(0,1).reshape([b, -1, t, f])
def conv2Dpad(outputs, shape):
    shape = list(shape)
    oshape = list(outputs.shape)
    if shape[2] != outputs.shape[2]:
        assert(shape[2] > outputs.shape[2])
        shape[1] = oshape[1] # set channel
        shape[2] = shape[2] - outputs.shape[2]
        shape[3] = oshape[3]
        try:
            assert(shape[2] < 1024)
        except:
            assert(shape[2] < 1024)
        concat = [outputs, nd.zeros(shape, outputs.context)]
        outputs = nd.concat(*concat, dim = 2)
    return outputs
