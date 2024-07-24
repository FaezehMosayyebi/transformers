import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class PatcheEmbedding(nn.Module):
    def __init__(self, image_size: tuple[int, int, int, int], patch_size: int, projection_dim: int):
        super(PatcheEmbedding, self).__init__()

        self.batch_size, self.channels, height, width = image_size
        self.patch_size = patch_size
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)

        self.projection = nn.Linear(self.channels*self.patch_size*self.patch_size, projection_dim)
        self.projection_embedding = nn.Embedding(self.num_patches, projection_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(self.batch_size, self.channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(self.batch_size, self.num_patches, -1)
        patches = self.projection(patches)

        positions = torch.arange(start=0, end = self.num_patches, step=1)[None,:]
        positional_embedding = self.projection_embedding(positions)

        patches = patches + positional_embedding
   
        return patches

    def get_config(self):
        return {"patch_size": self.patch_size}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test
    image = torchvision.io.read_image('images.jpg')
    image = image[None, :]
    patching = PatcheEmbedding(16)
    patches = patching(image)

    print(patches.shape)