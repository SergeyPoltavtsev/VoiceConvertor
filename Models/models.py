from Network.network import Network


class EVANET(Network):
    alpha = [0, 0, 0, 1, 1]
    beta = [1, 1, 1, 1, 1]

    def setup(self):
        (self.conv(3, 3, 2, 64, name='conv1_1')
         .conv(3, 3, 64, 64, name='conv1_2')
         .pool(name='pool1')
         .conv(3, 3, 64, 128, name='conv2_1')
         .conv(3, 3, 128, 128, name='conv2_2')
         .pool(name='pool2')
         .conv(3, 3, 128, 256, name='conv3_1')
         .conv(3, 3, 256, 256, name='conv3_2')
         .pool(name='pool3')
         )

    def y(self):
        return [self.vardict['pool3']]


def getModel(input, params_path):
    return EVANET(input, params_path)
