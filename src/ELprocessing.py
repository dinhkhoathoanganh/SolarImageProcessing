from util import *

# # User's configurations can be specified at *UserInput.py*

img_list = read_img_list()  ## Run all images in input folder
# img_list = ['image4'] ## To test selected images in input folder

print("IMAGES TO PROCESS IS: ")
print(img_list)

for img in img_list:
    create_output_folder(img)
    run_splicing_image(img)

print("DONE!")
