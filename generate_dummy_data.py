import os
import cv2
import numpy as np

def create_dummy_data():
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/masks', exist_ok=True)
    
    for i in range(1, 6):
        img_path = f'data/images/patient_{i:03d}.png'
        mask_path = f'data/masks/patient_{i:03d}_mask.png'
        
        # Create a dummy 256x256 image (random noise)
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        
        # Create a dummy 256x256 mask (random binary)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        cv2.imwrite(mask_path, mask)
        
    print("Dummy data generated successfully!")

if __name__ == '__main__':
    create_dummy_data()
