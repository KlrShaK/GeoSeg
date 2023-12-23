import os
import cv2
import torch
import numpy as np
from PIL import Image
from geoseg.models.UNetFormer import UNetFormer  # or replace with your model
from torchvision import transforms
from torch.utils.data import DataLoader
from geoseg.datasets.potsdam_dataset import PotsdamDataset, PALETTE

def load_model(model_path, num_classes):
    model = UNetFormer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_folder(input_folder, output_folder, model):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        save_segmented_image(output, output_folder, image_name, PALETTE)

def save_segmented_image(segmentation, output_folder, image_name, palette):
    # Convert segmentation to RGB
    segmented_img = Image.fromarray(np.uint8(segmentation), mode='P')
    segmented_img.putpalette([color for sublist in palette for color in sublist])
    segmented_img = segmented_img.convert('RGB')

    # Save image
    save_path = os.path.join(output_folder, image_name)
    segmented_img.save(save_path)

if __name__ == "__main__":
    input_folder = 'path/to/your/input/images'
    output_folder = 'path/to/save/segmented/images'
    model_path = 'path/to/your/model.pth'
    num_classes = 6  # Update based on your model

    model = load_model(model_path, num_classes)
    process_folder(input_folder, output_folder, model)
    print("Segmentation completed and saved in:", output_folder)
