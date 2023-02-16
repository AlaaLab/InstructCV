import torch
import jieba
from universal_mae_helpers import get_cuda_devices

class VisualCue(torch.nn.Module):

    def __init__(self, image_size):
        super().__init__()

        #self.visual_cue = torch.nn.init.xavier_uniform(torch.nn.Parameter(torch.zeros(1, image_size, int(image_size/2), 3)))

        self.encoder_layer  = torch.nn.TransformerEncoderLayer(d_model=256, nhead=4)

        self.word_embedding = torch.nn.Embedding(1000, 256, max_norm=True)  # the max number of lauguage words is 1000

        self.fc             = torch.nn.Linear(256, image_size*image_size*3//2)

        self.image_size     = image_size

    def forward(self, language_token):
        
        device              = get_cuda_devices()
        language_token      = language_token.to(device)
        embeddings          = self.word_embedding(language_token)
        output              = self.encoder_layer(embeddings)

        output = output[ :, -1: ]

        output = self.fc(output)

        visual_cue          = torch.reshape(output, (-1, self.image_size, int(self.image_size/2), 3))

        return visual_cue
