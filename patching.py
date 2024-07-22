import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

class Patches(nn.Module):
    def __init__(self, patch_size: int):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(batch_size, num_patches_h * num_patches_w, -1)
   
        return patches

    def get_config(self):
        return {"patch_size": self.patch_size}


if __name__ == "__main__":

    # test
    image = torchvision.io.read_image('images.jpg')
    image = image[None, :]
    patching = Patches(16)
    patches = patching(image)

    patchsize = patching.get_config()["patch_size"]
    for i , patch in enumerate(patches[0]):
        ax = plt.subplot(image.shape[2]//patchsize, image.shape[3]//patchsize, i+1)
        plt.imshow(patch.view(patchsize, patchsize, 3).numpy().astype("uint8"))
        plt.axis("off")
    plt.show()