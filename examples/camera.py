import pickle
from PIL import Image


with open(r"\Users\nicep\Desktop\image_multi.pkl", "rb") as f:
    image_data, raw_data = pickle.load(f)

img = Image.fromarray(image_data)
img.show()

print("hi")
