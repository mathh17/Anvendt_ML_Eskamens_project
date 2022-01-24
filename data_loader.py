#import packages for labelling and converting imagery data
import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray

##############
# Func for creating 1D image arrays

def load_image_function(path):

    images = [] # empty list placeholder
    labels = [] # empty list placeholder
    container = [] # container to validate correct labels

    for filename in os.listdir(path):

      container.append(filename) # add filename to container

      CCDY_img = load_img(path + f'/{filename}', 
                          target_size = (56, 106), 
                          color_mode="grayscale") # standardize photo size + loads
        
      CCDY_img = img_to_array(CCDY_img).flatten() # creates an array for imagery values

      images.append(CCDY_img) # append the photo to the images. The images list contains a list of arrays
    
    return asarray(images), container; # not interested in list of arrays, but array containing lists. Asarray does this. Returns three arrays


################
# Func for creating df with classes and 1d img arrays

def load_1d_grays ():
    # Start: creatign classes ons string_digits
    # load string digits
    #os.chdir(path_string_digits)
    string_digits = pd.read_csv('DIDA_12000_String_Digit_Labels.csv', 
                 header = None, 
                 names=["index", "string"])
    # create empty class columns
    string_digits['CC'] = 0
    string_digits['D'] = 0
    string_digits['Y'] = 0
    string_digits = string_digits.astype(str)
    # Iterate string digits and append classes
    for i, row in string_digits.iterrows():
        if len(row['string']) != 4:
            row['CC'] = '1'
            row['D'] = '10'
            row['Y'] = '10'
        else:
            row['D'] = row['string'][2]
            row['Y'] = row['string'][3]
            if row['string'][0:2] == '18':
                row['CC']='0'
            else:
                row['CC']='1'
    # End of class labeling on string_digits.
    #
    # Start: create img_df containing scaled images as 1D tensors
    # Convert imagery to 1D arrays with tagged file names
    #os.chdir(path_images)
    image_array, filename = load_image_function('DIDA_12000_String_Digit_Images\\DIDA_1')
    # and convert to a img_df
    img_df = pd.DataFrame({'filename': filename, 'gray_value': list(image_array)}, 
                          columns=['filename', 'gray_value'])
    # Create proper index value in img_df to allow merge on string_digits
    img_df['index'] = img_df['filename']
    for i, row in img_df.iterrows():
        row['index'] = str(img_df['index'][i]).split('.')[0]
    img_df
    # End of creating img_df
    #
    # Start: merge img_df with string_digits
    # match index type on dataframes to merge
    string_digits['index'] = string_digits['index'].astype(int)
    img_df['index'] = img_df['index'].astype(int)
    # Merge dataframes
    df_img_classes = string_digits.merge(img_df)
    # Rearrange order of dataframe
    df_img_classes = df_img_classes.reindex(columns= ['index', 'string', 'CC', 'D', 'Y', 'gray_value', 'filename'])
    # End og merging data frames
    return df_img_classes

