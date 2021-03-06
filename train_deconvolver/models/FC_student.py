from torchsummary import summary
import torch.nn as nn


class FC_student(nn.Module):

    def __init__(self, n_filters, k_conv=3, k_t_conv = 2, input_channels=1):
        super(FC_student, self).__init__()
        self.input_channels = input_channels
        self.n_filters = n_filters
        self.k_conv = k_conv
        self.k_t_conv = k_t_conv
        self.padding = k_conv // 2
        # non riduce le dimensioni spaziali, aumenta soltanto il numero di filtri
        self.conv1 = nn.Conv3d(input_channels,
                               n_filters, self.k_conv, padding=self.padding)
        self.conv2 = nn.Conv3d(self.n_filters,
                               self.n_filters*2, self.k_conv,
                               padding=self.padding)
        '''
        self.conv3 = nn.Conv3d(self.n_filters*2,
                               self.n_filters*4, self.k_conv,
                               padding=self.padding)
        '''

        # Deconvoluzioni con filtri 2x2x2 e stride 2 raddoppiano le dimensioni
        '''
        self.conv_t1 = nn.ConvTranspose3d(4 * n_filters, 2*n_filters,
                                          self.k_t_conv, stride=2)
        self.conv_t2 = nn.ConvTranspose3d(2*n_filters,
                                          n_filters, self.k_t_conv, stride=2)
        '''
        self.conv_t = nn.ConvTranspose3d(2*n_filters, n_filters, self.k_t_conv, stride=2)
        # 1x1x1 conv https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        self.conv1x1x1 = nn.Conv3d(self.n_filters, input_channels, 1)

        self.max_pool = nn.MaxPool3d(2)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        input = input.unsqueeze(1)

        output = self.conv1(input)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.conv2(output)
        output = self.relu(output)
        '''
        output = self.max_pool(output)
    
        output = self.conv3(output)
        output = self.relu(output)
        #output = self.max_pool(output)
        
        output = self.conv_t1(output)
        
        #output = self.relu(output)
        '''
        output = self.conv_t(output)
        output = self.relu(output)

        output = self.conv1x1x1(output)
        #output = self.sigmoid(output)

        return output.squeeze(1)

if __name__=="__main__":

    student = FC_student(4, k_conv=7, k_t_conv=2).to('cuda:0')
    summary(student, input_size=(64, 64, 64))
    #for i, p in enumerate(teacher.parameters()):
    #    print(i, p.shape)
    #    print (teacher._modules.keys()[i//2])
    #    print(p.grad)