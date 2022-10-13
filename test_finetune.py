from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import timm
import torch
from torch.layers import Input
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=1)

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]
print(classes[preds[0]])

m = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100)
# a = torch.randn(1, 3, 224, 224)
# b = m(a)
# print(b)
# print(b.shape)
input = Input(shape=(32,32, 3),name = 'image_input')
outputs = m(input)