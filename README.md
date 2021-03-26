# ImagesBlending
A program to blend 2 images using a mask.

How to use?
1. Download the files:
  - 'blender.py'
  - 'externals'
  - optional- 'requirments.txt'
2. To blend, write on the command line:
  (a) python blender.py <image1 path> <image2 path> <mask path> <param1> <param2> <param3>
  or:
  (b) python blender.py <image1 path> <image2 path> <mask path>
  where:
  - <image1 path> is the path to the first image (has to be '.jpg')
  - <image2 path> is the path to the second image (has to be '.jpg')
  - <mask path> is the path to the mask (has to be '.jpg', and black/white)
  - param1 is an integer in the range [1, 11] determine the strength of the blending
  - param2 is an odd integer in the range [1, 11] determine the filter size for the images
  - param3 is an odd integer in the range [1, 11] determine the filter size for the mask
  If you chose (b), the default parameter for param1-3 is 3.
3. To see some examples you can write on the command line:
  python blender.py example1
  or:
  python blender.py example2
  
Have fun!
