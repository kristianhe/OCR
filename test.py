from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img("images/a_0.jpg")
image = img_to_array(image)
image = np.expand_dims(image, axis=0)



for path, _, files in os.walk("chars74k-lite"):
    print(path)


total = 10

# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(
	rotation_range=180)
	#rescale = 1./255)
total = 0

# construct the actual Python generator
print("> generating augmented images for training data extension...")
imageGen = aug.flow(image, batch_size=1, save_to_dir="augmented-images",
	save_prefix="image", save_format="jpg")
 
# loop over examples from our image data augmentation generator
for image in imageGen:
	# increment our counter
	total += 1
	# if we have reached the specified number of examples, break
	if total == total:
		break
