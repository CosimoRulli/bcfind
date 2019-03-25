from torchsummary import summary
import torch.nn as nn


class FC_teacher_max_p(nn.Module):
    def __init__(self, n_filters, input_channels=1, k_conv=3, k_t_conv = 2):
        super(FC_teacher_max_p, self).__init__()
        self.input_channels = input_channels
        self.n_filters = n_filters
        self.k = k_conv
        #non riduce le dimensioni spaziali, aumenta soltanto il numero di filtri
        self.conv1 = nn.Conv3d(input_channels, n_filters, self.k, padding=1)
        self.conv2 = nn.Conv3d(self.n_filters, self.n_filters*2, self.k, padding=1)
        self.conv3 = nn.Conv3d(self.n_filters*2, self.n_filters*4, self.k, padding=1)


        #Deconvoluzioni con filtri 2x2x2 e stride 2 raddoppiano le dimensioni
        self.conv_t1 = nn.ConvTranspose3d(4 * n_filters, 2*n_filters,
                                          k_t_conv, stride=2)
        self.conv_t2 = nn.ConvTranspose3d(2*n_filters, n_filters, k_t_conv, stride=2)

        #1x1x1 conv https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        self.conv1x1x1= nn.Conv3d(self.n_filters, input_channels, 1)

        self.max_pool = nn.MaxPool3d(2)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        output = self.conv1(input)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv3(output)
        output = self.relu(output)
        #output = self.max_pool(output)

        output = self.conv_t1(output)
        #output = self.relu(output)
        output = self.conv_t2(output)
        output = self.relu(output)



        output = self.conv1x1x1(output)
        output = self.sigmoid(output)

        return output

if __name__=="__main__":
    teacher = FC_teacher_max_p(4).to('cuda:0')
    summary(teacher, input_size=(1,64,64,64))