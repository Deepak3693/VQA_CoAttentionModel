import os
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

currentDirectory = os.getcwd()
dataDirectory = currentDirectory + "Data/"
train_imageDirectory = dataDirectory + "Train/train2014/"
train_question_file_path = dataDirectory + 'Train/v2_OpenEnded_mscoco_train2014_questions.json'
train_annotation_file_path = dataDirectory + 'Train/v2_mscoco_train2014_annotations.json'
validation_imageDirectory = dataDirectory + "Validation/val2014/"
validation_question_file_path = dataDirectory + 'Validation/v2_OpenEnded_mscoco_val2014_questions.json'
validation_annotation_file_path = dataDirectory + 'Validation/v2_mscoco_val2014_annotations.json'
test_imageDirectory = dataDirectory + "Test/test2015/"
test_question_file_path = dataDirectory + 'Test/v2_OpenEnded_mscoco_test2015_questions.json'

#Data Transformation

os.chdir(dataDirectory + "Train/")
with open(train_question_file_path, 'r') as f:
    train_questions = json.load(f)
    train_questions = train_questions["questions"]

with open(train_annotation_file_path, 'r') as f:
    train_annotations = json.load(f)
    train_annotations = train_annotations["annotations"]

os.chdir(dataDirectory + "Validation/")
with open(validation_question_file_path, 'r') as f:
    validation_questions = json.load(f)
    validation_questions = validation_questions["questions"]

with open(validation_annotation_file_path, 'r') as f:
    validation_annotations = json.load(f)
    validation_annotations = validation_annotations["annotations"]
 
# Overview of extracted Data
    
print("Total Number Questions in Train Dataset are : ",len(train_questions))
train_questions_df = pd.DataFrame(train_questions)
print(train_questions_df.head(5))

train_annotations_df = pd.DataFrame(train_annotations)
print(train_annotations_df.head(5))

print("Total Number Questions in Validation Dataset are : ",len(validation_questions))
validation_questions_df = pd.DataFrame(validation_questions)
print(validation_questions_df.head(5))

validation_annotations_df = pd.DataFrame(validation_annotations)
print(validation_annotations_df.head(5))

train_data = pd.merge(train_questions_df,train_annotations_df,  how='inner', left_on=['image_id','question_id'], right_on = ['image_id','question_id'])
print(train_data.head(5))

validation_data = pd.merge(validation_questions_df,validation_annotations_df,  how='inner', left_on=['image_id','question_id'], right_on = ['image_id','question_id'])
print(validation_data.head(5))


# sample image
os.chdir(train_imageDirectory)
index = np.random.randint(0,len(train_data))#263115

img_path =  train_imageDirectory + 'COCO_train2014_' + '%012d.jpg' % (train_data['image_id'][index])
img=mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
print("*"*50)
print("Question : " ,train_data['question'][index])
print("*"*50)
print("Answer : ", train_data['multiple_choice_answer'][index])

print(img.shape)

os.chdir(validation_imageDirectory)
index = np.random.randint(0,len(validation_data))#263115

img_path =  validation_imageDirectory + 'COCO_val2014_' + '%012d.jpg' % (validation_data['image_id'][index])
img=mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()
print("*"*50)
print("Question : " ,validation_data['question'][index])
print("*"*50)
print("Answer : ", validation_data['multiple_choice_answer'][index])

print(img.shape)

#Saving Final Data
train_data.to_csv(dataDirectory + 'train_data.csv')
validation_data.to_csv(dataDirectory + 'validation_data.csv')