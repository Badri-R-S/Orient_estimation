import argparse
import torch
import os
import cv2
import supervision as sv
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(script_dir, "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'
print(DEVICE)
MODEL_TYPE = "vit_h"
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
#Automted Mask generation
mask_generator = SamAutomaticMaskGenerator(sam)
def mask_image(IMAGE_PATH):
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_result = mask_generator.generate(image_rgb)

    #print(sam_result[0].keys())

    ## VIEW MASKS ###
    #masks = [
    #    mask['segmentation']
    #    for mask
    #    in sorted(sam_result, key=lambda x:x['area'], reverse=True)
    #]
    
    #sv.plot_images_grid(
    #    images = masks,
    #    grid_size= (32,1),
    #    size=(32,32)
    #)
    #print(sam_result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    mask = ((sam_result[1]['segmentation'])*255).astype(np.uint8)

    masked_image = cv2.bitwise_and(image_rgb,image_rgb, mask=mask)
    return masked_image
    

if __name__ == '__main__':
    
    # Path to template and test image folders
    template_folder = "/home/badri/mowito/template_images"
    test_folder = "/home/badri/mowito/test_images"
    i = 1
    j = 1
    
    for template_filename in sorted(os.listdir(template_folder)):
        template_path = os.path.join(template_folder, template_filename)
        masked_image = mask_image(template_path)
        output_path = os.path.join(template_folder, 'masked_template_'+str(i)+'.jpg')
        cv2.imwrite(output_path, masked_image)
        i = i+1

    # Iterate over the subdirectories inside the test_images directory
    for type_folder in os.listdir(test_folder):
        type_folder_path = os.path.join(test_folder, type_folder)
        
        # Check if the item is a directory
        if os.path.isdir(type_folder_path):
            # Iterate over the images inside each type folder
            for i in range(1, 6):  # Assuming there are 5 images numbered from 1.jpg to 5.jpg
                image_path = os.path.join(type_folder_path, f"{i}.jpg")
                
                # Check if the image file exists
                if os.path.exists(image_path):
                    # Read the image using OpenCV
                    masked_image = mask_image(image_path)
                    output_path = os.path.join(type_folder_path, 'test_masked'+str(i)+'.jpg')
                    cv2.imwrite(output_path, masked_image)
    

