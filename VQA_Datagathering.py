# -*- coding: utf-8 -*-
#importing packages

import numpy as np
import os
import tensorflow as tf
import json

#Assigning variables which are used in this project
currentDirectory = os.getcwd()
currentDirectory = currentDirectory + "Data/"
os.chdir(currentDirectory)

#Downlaoding and Extracting Images into Train folder
os.chdir(currentDirectory + "Train/")
tf.keras.utils.get_file('train2014.zip', cache_subdir = os.path.abspath('.'), 
                        origin = 'http://images.cocodataset.org/zips/train2014.zip', extract = True)

# Displaying the total Number Images in COCO Train Dataset
os.chdir(currentDirectory + 'Train/train2014/')
print("Total Number Images in COCO Train Dataset: ",len([name for name in os.listdir()]))

#Downlaoding and Extracting Questions into Train folder
os.chdir(currentDirectory + "Train/")
tf.keras.utils.get_file('v2_Questions_Train_mscoco.zip',cache_subdir=os.path.abspath('.'),
                        origin = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',extract = True)

# Read the  Questions json file
question_file_path = 'v2_OpenEnded_mscoco_train2014_questions.json'
with open(question_file_path, 'r') as f:
    questions = json.load(f)

print("Total Number Questions is : ",len(questions['questions']))
print(questions['questions'][np.random.randint(0,443757)])

#Downlaoding and Extracting annotations
tf.keras.utils.get_file('v2_Annotations_Train_mscoco.zip',cache_subdir=os.path.abspath('.'),
                        origin = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',extract = True)

# Reading annations json file
annotation_file_path = 'v2_mscoco_train2014_annotations.json'
with open(annotation_file_path, 'r') as f:
    annotations = json.load(f)

print(annotations['annotations'][np.random.randint(0,443757)])

#Downlaoding and Extrcating Images into Validation folder
os.chdir(currentDirectory + "Validation/")
tf.keras.utils.get_file('train2014.zip', cache_subdir = os.path.abspath('.'), 
                        origin = 'http://images.cocodataset.org/zips/val2014.zip', extract = True)

# Displaying total Number of Images in COCO Validation Dataset
os.chdir(currentDirectory + 'Validation/val2014/')
print("Total Number Images in COCO Validation Dataset: ",len([name for name in os.listdir()]))

#Downlaoding and Extracting Questions into Validation folder
os.chdir(currentDirectory + "Validation/")
tf.keras.utils.get_file('v2_Questions_Val_mscoco.zip',cache_subdir=os.path.abspath('.'),
                        origin = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',extract = True)

# read the validations question json file
val_question_file_path = 'v2_OpenEnded_mscoco_val2014_questions.json'
with open(val_question_file_path, 'r') as f:
    val_questions = json.load(f)

print("Total Number Questions is : ",len(val_questions['questions']))
print(val_questions['questions'][np.random.randint(0,443757)])

#Downlaoding and Extracting annotations
tf.keras.utils.get_file('v2_Annotations_Val_mscoco.zip',cache_subdir=os.path.abspath('.'),
                        origin = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',extract = True)

# Reading the validation annotations json file
val_annotation_file_path = 'v2_mscoco_val2014_annotations.json'
with open(val_annotation_file_path, 'r') as f:
    val_annotations = json.load(f)
    
print(val_annotations['annotations'][np.random.randint(0,443757)])

#Downlaoding and Extrcating Images into Test folder
os.chdir(currentDirectory + "Test/")
tf.keras.utils.get_file('test2015.zip', cache_subdir = os.path.abspath('.'), 
                        origin = 'http://images.cocodataset.org/zips/test2015.zip', extract = True)

# Displaying Total Number of Images in COCO Train Dataset
os.chdir(currentDirectory + 'Test/test2015/')
print("Total Number Images in COCO Train Dataset: ",len([name for name in os.listdir()]))

#Downlaoding and Extracting Questions into Validation folder
os.chdir(currentDirectory + "Test/")
tf.keras.utils.get_file('v2_Questions_Test_mscoco.zip',cache_subdir=os.path.abspath('.'),
                        origin = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip',extract = True)

# read the questions json file
test_question_file_path = 'v2_OpenEnded_mscoco_test2015_questions.json'
with open(test_question_file_path, 'r') as f:
    test_questions = json.load(f)

print("Total Number Questions is : ",len(test_questions['questions']))

print(test_questions['questions'][np.random.randint(0,443757)])