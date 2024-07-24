import cv2
import matplotlib.pyplot as plt

# Load your image
img = cv2.imread('images\cat image.jfif')  # Replace with your image path

# Check if the image is loaded properly
if img is None:
    print("Error: Could not load image.")
else:
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide the axes
    plt.show()
