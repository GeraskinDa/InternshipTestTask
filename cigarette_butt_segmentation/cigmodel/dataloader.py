from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class CigDataset(Dataset):
    """Dataset class for DataLoader generator

    Attributes
    ----------
    img_names : list[str]
        names of images in the folder path/images
    mask_names : list[str]
        names of masks in the folder path/masks
    weight_names : list[str]
        names of weights in the folder path/masks

    """

    def __init__(self, path, augment=False, augm_only=False):
        """
        The constructor for CigDataset class

        Parameters
        ----------
        path : str
            Path to folder with images, masks and weights.
        augment : bool
            If true, uses augmented data with main data
        augm_only : bool
            If augm_only is true, uses only augmented data

        Notes
        -----
        If augm_only is True, use path/augment instead

        """
        self.path = path
        self.img_names = sort_names(f"{self.path}/images")
        self.mask_names = sort_names(f"{self.path}/masks")
        self.weight_names = sort_names(f"{self.path}/weights")
        self.trans_to_tens = transforms.ToTensor()
        self.trans_to_norm = transforms.Normalize([0.5315, 0.5226, 0.4595],
                                                  [0.2005, 0.2006, 0.1882])
        self.augm_only = augm_only

        if augm_only is False:
            if augment:
                self.img_names.extend(
                    sort_names(f"{self.path}/augment/images"))
                self.mask_names.extend(
                    sort_names(f"{self.path}/augment/masks"))
                self.weight_names.extend(
                    sort_names(f"{self.path}/augment/weights"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        """
        Method for returning samples

        Returns
        -------
        dict
            Image and corresponding mask and weight mask

        """
        if self.augm_only is False:
            if self.img_names[item][0:3] == 'aug':
                img = self.trans_to_norm(self.trans_to_tens(
                    Image.open(f"{self.path}/augment/images/"
                               f"{self.img_names[item]}").convert('RGB')))

                mask = self.trans_to_tens(Image.open(f"{self.path}/augment/masks/"
                                                     f"{self.mask_names[item]}"))
                weight_mask = self.trans_to_tens(Image.open(f"{self.path}/augment/weights/"
                                                            f"{self.weight_names[item]}"))

            else:
                img = self.trans_to_norm(self.trans_to_tens(
                    Image.open(f"{self.path}/images/"
                               f"{self.img_names[item]}").convert('RGB')))

                mask = self.trans_to_tens(Image.open(f"{self.path}/masks/"
                                                     f"{self.mask_names[item]}"))
                weight_mask = self.trans_to_tens(Image.open(f"{self.path}/weights/"
                                                            f"{self.weight_names[item]}"))
        else:
            img = self.trans_to_norm(self.trans_to_tens(
                Image.open(f"{self.path}/images/"
                           f"{self.img_names[item]}").convert('RGB')))

            mask = self.trans_to_tens(Image.open(f"{self.path}/masks/"
                                                 f"{self.mask_names[item]}"))
            weight_mask = self.trans_to_tens(Image.open(f"{self.path}/weights/"
                                                        f"{self.weight_names[item]}"))

        return {'image': img, 'mask': mask, 'weight': weight_mask}


def sort_names(path):
    """
    Returns sorted names in the folder at the specified path

    Parameters
    ----------
    path : str
        Path to the folder

    Returns
    -------
    list
        Sorted list of the names in the folder

    """
    return sorted(os.listdir(f"{path}"))
