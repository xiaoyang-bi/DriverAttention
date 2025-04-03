import sys
import PIL
from pathlib import Path

import PIL.Image
import numpy as np
from PIL import Image


def image_to_array(image):
    return np.array(image, dtype=np.float32)

def atten_mixup(imga, imgb, attena, attenb):
    imga = image_to_array(imga)
    imgb = image_to_array(imgb)
    attena = image_to_array(attena)
    attenb = image_to_array(attenb)
    
    attena = np.expand_dims(attena, axis=-1)
    attenb = np.expand_dims(attenb, axis=-1)
    attena /= 255.
    attenb /= 255.
    # print(attena.shape)
    # print()
    eps = 1e-7
    
    mix_data = (imga * (attena+eps) + imgb * (attenb + eps)) / (attena + attenb + 2*eps)
    return Image.fromarray(np.uint8(mix_data))

def trival_mixup(imga, imgb, alpha=1.):
    '''random select a data in batch and mix it
    attention is better not divide the max
    '''
    imga = image_to_array(imga)
    imgb = image_to_array(imgb)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mix_data = lam * imga + (1 - lam) * imgb
    return Image.fromarray(np.uint8(mix_data))


def ensure_mode(image, mode='RGB'):
    if image.mode != mode:
        image = image.convert(mode)
    return image

if __name__ == '__main__':
    roota = sys.argv[1]
    rootb = sys.argv[2]
    exa = sys.argv[3]
    exb = sys.argv[4]

    camera_dir = 'camera'
    gaze_dir = 'infer_gaze'

    # Resize the attention maps to the size of the example images


    attena = PIL.Image.open(Path(roota) / gaze_dir / exa)
    attenb = PIL.Image.open(Path(rootb) / gaze_dir / exb)
    
    exa = PIL.Image.open(Path(roota) / camera_dir / exa)
    exb = PIL.Image.open(Path(rootb) / camera_dir / exb)
    
    attena = attena.resize(exa.size, PIL.Image.BILINEAR)
    attenb = attenb.resize(exb.size, PIL.Image.BILINEAR)
    
    exa = ensure_mode(exa, 'RGB')
    exb = ensure_mode(exb, 'RGB')
    attena = ensure_mode(attena, 'L')  # Assuming attention maps are grayscale
    attenb = ensure_mode(attenb, 'L')

    attenmix_img = atten_mixup(exa, exb, attena, attenb)
    trivalmix_img = trival_mixup(exa, exb)

    # Save the attenmix_img and trivalmix_img to disk
    attenmix_img.save(Path('./') / 'mixup_demo' / 'attenmix_image1.png')
    trivalmix_img.save(Path('./') / 'mixup_demo' / 'trivalmix_image1.png')
