import os
from PIL import Image
import warnings
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

dir = 'C:/Users/evank/OneDrive/Documents/HALnalysis_summer_2024_data/plots/M07480'

cwd = os.getcwd()
target_dir = os.path.join(cwd, 'temp_chip_vis_images')

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

files_to_process = []
for root, _, files in os.walk(dir):
    for file in files:
        if file == 'chip_vis.png':
            files_to_process.append((root, file))

for root, file in tqdm(files_to_process, desc="Processing images"):
    source_file = os.path.join(root, file)
    relative_path = os.path.relpath(root, dir)
    new_file_name = relative_path.replace(os.sep, '.') + '.chip_vis.png'
    target_file = os.path.join(target_dir, new_file_name)
    
    with Image.open(source_file) as img:
        new_size = (int(img.width * 0.2), int(img.height * 0.2))
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(target_file)



# this script collects all images named 'chip_vis.png' in the specified directory and resizes them to 20% of their original size
# this makes it really easy to import them into GIMP and create a GIF

# how to:
# run this script on the chip file structure
# install GIMP
# install G'MIC plugin: https://gmic.eu/download.html
# run .exe and restart computer
# in GIMP, import all images as layers
# take the eye icon off all the layers except the ones you want in one frame for the gif
# go to Filters -> G'MIC -> Montage
# select vertical and on the bottom set input to all visible layers
# click okay - those layers will be merged into one layer
# repeat for all frames
# then go to File -> Export As -> name the file as montage.gif and then set these (recommended) options: 
# animation, 1500 ms inbetween frames, frame disposal (replace), 'use delay entered above for all frames' and 'use disposal entered above for all frames'
# don't use 'use delay entered above for all frames' if you want to specify the duration of each frame manually
# don't use the optimize for gif option if using the replace frame disposal option - check: https://www.reddit.com/r/GIMP/comments/xr4xp8/help_with_gimp_gif_layers_being_distorted_on/
# the unoptimize button doesn't really work, so if you optimized it, you'll have to start over and make a new project 😔
# program will go unresponsive - it's okay, check task manager to see it working - it'll eventually finish

# if you are adding text to the gifs, make sure to merge all the visible frames with CRTL + M - this will merge all visible frames into one frame
# you can manually specify the duration of each frame in the gif by adding (200ms) to the layer name
# and you can add the either (combine) or (replace) to the layer name to specify how the layer should be merged with the previous frame
# so if you do add (2000ms)(replace) to the layer name, make sure to uncheck those last two options in the export dialog



