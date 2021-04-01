# ImagesBlending
A program to blend 2 images using a mask.

How to use?
1. Download the files:
    - 'blender.py'
    - 'externals'
    - optional- 'requirments.txt'
2. To blend, write on the command line:
 
    (a) python blender.py IMAGE1PATH IMAGE2PATH MASK_PATH PARAM1 PARAM2 PARAM3
  
    (b) python blender.py IMAGE1PATH IMAGE2PATH MASK_PATH
  
    where:
    - IMAGE1PATH is the path to the first image (has to be '.jpg')
    - IMAGE2PATH is the path to the second image (has to be '.jpg')
    - MASK_PATH is the path to the mask (has to be '.jpg', and black/white)
    - PARAM1 is an integer in the range [1, 11] determine the strength of the blending
    - PARAM2 is an odd integer in the range [1, 11] determine the filter size for the images
    - PARAM3 is an odd integer in the range [1, 11] determine the filter size for the mask
    
    If you chose (b), the default parameter for PARAM1-3 is 3.
3. To see some examples you can write on the command line:

    python blender.py example1
    
    or:
    
    python blender.py example2
  
Have fun!
