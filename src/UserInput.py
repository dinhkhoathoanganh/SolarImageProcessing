# Image file
file_extension = '.jpg'

# Folder path
# Folder path must contain 1 folder named "input" = Containing raw images | "output" folder will be auto-generated
image_folder = 'C:/Users/dinhk_1phzc8w/Desktop/etavolt/data/'

# Save all splices' images
# True = Save | False = Not Save, just trial run
verbose = True

# Image height, width (see as horizontally)
WIDTH = 850
HEIGHT = 535
MIN_CELL_AREA = 4000
MAX_CELL_AREA = 10000
SPLICES = (6, 10)                # No cells in module (y,x)
ERROR_MARGIN_SPLICING = 0.4      # X% of error allowed when cell splicing (1 +- X)
