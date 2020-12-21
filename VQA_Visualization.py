# -*- coding: utf-8 -*-
#importing packages
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from wordcloud import WordCloud

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


#Data Visualization on train_data
os.chdir(dataDirectory)
train_data = pd.read_csv(dataDirectory + 'train_data.csv')
validation_data = pd.read_csv(dataDirectory + 'validation_data.csv')

#Box Plot

aggregations = {'question': 'count'}
temp = pd.DataFrame(train_data.groupby(['image_id'],as_index=False).agg(aggregations))
num_of_ques_in_image = temp['question'].values
print("Maximum number of questions:",max(num_of_ques_in_image))
print("Minimum number of questions:",min(num_of_ques_in_image))
print("Mean:",np.mean(num_of_ques_in_image))

ax = sns.boxplot(y = 'question', data = temp) 
plt.title("boxplot of Number of questions on a image")
plt.show()

os.chdir(train_imageDirectory)
#Image having most number of questions
image_path =  train_imageDirectory + 'COCO_train2014_' + '%012d.jpg' % (temp[temp['question'] == 275]['image_id'].values[0])
image=mpimg.imread(image_path)
imgplot = plt.imshow(image)
plt.axis('off')
plt.title("Image having maximum number of questions")
plt.show()

#Duplicate Questions on same Image
aggregations = {'question_id':'count', 'multiple_choice_answer': lambda x: " || ".join(x)}
temp = pd.DataFrame(train_data.groupby(['image_id','question'],as_index=False).agg(aggregations)).rename(columns={'question_id':'count'})
temp = temp[temp['count']>1]
print(temp)

#Visualizations on Questions

#Question Type
print(train_data.question_type.unique())

#Printing the Unique Questions
print("Number of unique Question type in dataset : ",len(train_data.question_type.unique()))


def getFrequnctDict(train_data,column,isJoin=False):
    column_frequency = {}

    for _row in train_data[column]:
        if isJoin:
            _row = "_".join(_row.split())
        if(column_frequency.get(_row,-1) > 0):
            column_frequency[_row] += 1
        else:
            column_frequency[_row] = 1

    return column_frequency

def lineChart(train_data,column,top=20,isJoin=False):
    column_frequncy = getFrequnctDict(train_data,column,isJoin)
    sort_column_frequncy = sorted(list(column_frequncy.items()),key = lambda x: x[1],reverse=True)
    total_samples =  len(train_data)

    plt.plot([x[1]for x in sort_column_frequncy[:top]])
    i=np.arange(top)
    plt.title("Frequency of top " + str(top) + " " + column )
    plt.xlabel("Tags")
    plt.ylabel("Counts")
    plt.xticks(i,[x[0] for x in sort_column_frequncy[:top]])
    plt.xticks(rotation=90)
    plt.show()
    return sort_column_frequncy

def plotWordCloud(train_data,column,isJoin=False):
    column_frequncy = getFrequnctDict(train_data,column,isJoin)
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = None, 
                    min_font_size = 10).generate_from_frequencies(column_frequncy)
    # plot the WordCloud image     
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title("WordCloud on "+ column)  
    plt.show()

#plot Between WordCloud and Linechart
plotWordCloud(train_data, 'question_type')
question_type_frequncy = lineChart(train_data, 'question_type', top = 30)

# Different Question types in the dataset
for _type,_count in question_type_frequncy[:10]:
    print("Percentage of '" + _type + "' Type of Questions in Dataset is ", str(100*_count/len(train_data)) )
    
#Using Box Plot 
sns.countplot(train_data["question"].apply(lambda x: len(x.split())).values)
plt.title("Length of the questions vs Distrubution")
plt.xlabel("Length of the questions")
plt.ylabel("Distrubution")
plt.show()

##Visualization On Answers

#Unique Answers
train_data['answer_type'].unique()

#Line Chart for Answer_type
answer_type_frequncy = lineChart(train_data, 'answer_type', top = 3)

#Types of Answers in Dataset like 'Other', 'Yes/No', 'Number'
for _type,_count in answer_type_frequncy:
    print("Percentage of '" + _type + "' Type of Answers in Dataset is ", str(100*_count/len(train_data)) )
    
#Box Plot 
sns.countplot(train_data["multiple_choice_answer"].apply(lambda x: len(x.split())).values)
plt.title("Number of words in Answers vs Distrubution")
plt.xlabel("Number of words in Answers")
plt.ylabel("Distrubution")
plt.show()

#Visualization on Question Type vs Answer
fig = plt.figure(figsize=(80,30))
fig.tight_layout() 
count = 1
colorCodes = [ "#E18719", "#287819", "#2D6EAA", "#E6AF23", "#666666","#724D8D", "#EAAB5E", "#73A769","#93785F",
              "#C97B7B", "#81A8CC", "#EDC765", "#858585","#957AA9", "#F3CFA3","#B4D0AF", "#BEADA0", "#E4BDBD", 
              "#ABC5DD", "#F4DB9C", "#A3A3A3"]

for _type,_ in question_type_frequncy[:12]:

    percentage = str(round((len(train_data[train_data['question_type']==_type])/len(train_data))*100,1))+'%'

    plt.subplot(4, 3, count)
    temp = train_data[train_data['question_type']==_type]
    ax = temp['multiple_choice_answer'].value_counts()[:10][::-1].plot(kind='barh', figsize=(20,15),color=colorCodes[count-1], fontsize=13)
    ax.set_alpha(0.8)   
    ax.set_title("Question Type:  '" + _type + "' (" + percentage + ") vs Answer" , fontsize=18)
    ax.set_ylabel("Answers", fontsize=18)
    ax.get_xaxis().set_visible(False)


    for i in ax.patches:
        ax.text(i.get_width()/2, i.get_y(), str(round((i.get_width()/len(temp))*100, 2))+'%' + "(" +
                str(round((i.get_width()/len(train_data))*100, 2))+'%' +")", fontsize=10,color='black')
        
    count += 1

fig.tight_layout()
plt.show()


#Visualization on Answer vs Question type
fig = plt.figure()
fig.tight_layout() 
count = 1

colorCodes = [ "#E18719", "#287819", "#2D6EAA", "#E6AF23", "#666666","#724D8D", "#EAAB5E", "#73A769","#93785F",
              "#C97B7B", "#81A8CC", "#EDC765", "#858585","#957AA9", "#F3CFA3","#B4D0AF", "#BEADA0", "#E4BDBD", 
              "#ABC5DD", "#F4DB9C", "#A3A3A3"]

answer_frequncy = sorted(list(getFrequnctDict(train_data,'multiple_choice_answer').items()),key = lambda x: x[1],reverse=True)

for _type,_ in answer_frequncy[:12]:

    percentage = str(round((len(train_data[train_data['multiple_choice_answer']==_type])/len(train_data))*100,1))+'%'

    plt.subplot(4, 3, count)
    temp = train_data[train_data['multiple_choice_answer']==_type]
    ax = temp['question_type'].value_counts()[:10][::-1].plot(kind='barh', figsize=(20,15),color=colorCodes[count-1], fontsize=13)
    ax.set_alpha(0.8)   
    ax.set_title("Answer: '" + _type + "' (" + percentage + ") vs Question Type" , fontsize=18)
    ax.set_ylabel("Question Type", fontsize=18)
    ax.get_xaxis().set_visible(False)

    for i in ax.patches:
        ax.text(i.get_width()/2, i.get_y(), str(round((i.get_width()/len(temp))*100, 2))+'%' + "(" +
                str(round((i.get_width()/len(train_data))*100, 2))+'%' +")", fontsize=14,color='black')
        
    count += 1

fig.tight_layout()
plt.show()

#checking if acutal answer is same as persons answers

def getPeopleAnswer(answers):
    answers_dict = {}
    score_dict = { 'yes' : 3, 'maybe' : 2, 'no' : 1 }
    for _answer in answers:
        score = score_dict[_answer['answer_confidence']]
        if answers_dict.get(_answer['answer'],-1) != -1 :
            answers_dict[_answer['answer']] += score
        else:
            answers_dict[_answer['answer']] = score

    return sorted(list(answers_dict.items()),key = lambda x: x[1],reverse=True)[0][0]

train_data['derived_answer'] =  train_data["answers"].apply(lambda x: getPeopleAnswer(x))

#checking if Questions has any multiple answers
print(train_data[ train_data['derived_answer'] != train_data['multiple_choice_answer']])

##Data Visualization on Validation set
#Question Type
print(validation_data.question_type.unique())

#Box Plot

aggregations = {'question': 'count'}
temp = pd.DataFrame(validation_data.groupby(['image_id'],as_index=False).agg(aggregations))
num_of_ques_in_image = temp['question'].values
print("Maximum number of questions:",max(num_of_ques_in_image))
print("Minimum number of questions:",min(num_of_ques_in_image))
print("Mean:",np.mean(num_of_ques_in_image))

ax = sns.boxplot(y = 'question', data = temp) 
plt.title("boxplot of Number of questions on a image")
plt.show()

#Duplicate Questions on same Image
aggregations = {'question_id':'count', 'multiple_choice_answer': lambda x: " || ".join(x)}
temp = pd.DataFrame(train_data.groupby(['image_id','question'],as_index=False).agg(aggregations)).rename(columns={'question_id':'count'})
temp = temp[temp['count']>1]
print(temp)

#Visualization on Question

#Question Type
print(validation_data.question_type.unique())

#Printing the Unique Questions
print("Number of unique Question type in dataset : ",len(validation_data.question_type.unique()))

#plot Between WordCloud and Linechart
plotWordCloud(validation_data, 'question_type')
question_type_frequncy = lineChart(validation_data, 'question_type', top = 10)

for _type,_count in question_type_frequncy[:10]:
    print("Percentage of '" + _type + "' Type of Questions in Dataset is ", str(100*_count/len(validation_data)) )
    
#Using Box Plot 
sns.countplot(validation_data["question"].apply(lambda x: len(x.split())).values)
plt.title("Length of the questions vs Distrubution")
plt.xlabel("Length of the questions")
plt.ylabel("Distrubution")
plt.show()

#Visualization on Answers

#Unique Answers
print(validation_data['answer_type'].unique())

#Line Chart for Answer_type
answer_type_frequncy = lineChart(validation_data, 'answer_type', top = 3)
#Types of Answers in Dataset like 'Other', 'Yes/No', 'Number'
for _type,_count in answer_type_frequncy:
    print("Percentage of '" + _type + "' Type of Answers in Dataset is ", str(100*_count/len(validation_data)) )
    
#Box Plot 
sns.countplot(validation_data["multiple_choice_answer"].apply(lambda x: len(x.split())).values)
plt.title("Number of words in Answers vs Distrubution")
plt.xlabel("Number of words in Answers")
plt.ylabel("Distrubution")
plt.show()

#Visualization on Question Type and Answer
# Visualization on Questions Type vs Answer
fig = plt.figure(figsize=(80,30))
fig.tight_layout() 
count = 1
colorCodes = [ "#E18719", "#287819", "#2D6EAA", "#E6AF23", "#666666","#724D8D", "#EAAB5E", "#73A769","#93785F",
              "#C97B7B", "#81A8CC", "#EDC765", "#858585","#957AA9", "#F3CFA3","#B4D0AF", "#BEADA0", "#E4BDBD", 
              "#ABC5DD", "#F4DB9C", "#A3A3A3"]

for _type,_ in question_type_frequncy[:12]:

    percentage = str(round((len(validation_data[validation_data['question_type']==_type])/len(validation_data))*100,1))+'%'

    plt.subplot(4, 3, count)
    temp = validation_data[validation_data['question_type']==_type]
    ax = temp['multiple_choice_answer'].value_counts()[:10][::-1].plot(kind='barh', figsize=(20,15),color=colorCodes[count-1], fontsize=13)
    ax.set_alpha(0.8)   
    ax.set_title("Question Type:  '" + _type + "' (" + percentage + ") vs Answer" , fontsize=18)
    ax.set_ylabel("Answers", fontsize=18)
    ax.get_xaxis().set_visible(False)


    for i in ax.patches:
        ax.text(i.get_width()/2, i.get_y(), str(round((i.get_width()/len(temp))*100, 2))+'%' + "(" +
                str(round((i.get_width()/len(validation_data))*100, 2))+'%' +")", fontsize=10,color='black')
        
    count += 1

fig.tight_layout()
plt.show()

#Visulaizations on Answer vs Question Type
fig = plt.figure()
fig.tight_layout() 
count = 1

colorCodes = [ "#E18719", "#287819", "#2D6EAA", "#E6AF23", "#666666","#724D8D", "#EAAB5E", "#73A769","#93785F",
              "#C97B7B", "#81A8CC", "#EDC765", "#858585","#957AA9", "#F3CFA3","#B4D0AF", "#BEADA0", "#E4BDBD", 
              "#ABC5DD", "#F4DB9C", "#A3A3A3"]

answer_frequncy = sorted(list(getFrequnctDict(validation_data,'multiple_choice_answer').items()),key = lambda x: x[1],reverse=True)

for _type,_ in answer_frequncy[:12]:

    percentage = str(round((len(validation_data[validation_data['multiple_choice_answer']==_type])/len(validation_data))*100,1))+'%'

    plt.subplot(4, 3, count)
    temp = validation_data[validation_data['multiple_choice_answer']==_type]
    ax = temp['question_type'].value_counts()[:10][::-1].plot(kind='barh', figsize=(20,15),color=colorCodes[count-1], fontsize=13)
    ax.set_alpha(0.8)   
    ax.set_title("Answer: '" + _type + "' (" + percentage + ") vs Question Type" , fontsize=18)
    ax.set_ylabel("Question Type", fontsize=18)
    ax.get_xaxis().set_visible(False)

    for i in ax.patches:
        ax.text(i.get_width()/2, i.get_y(), str(round((i.get_width()/len(temp))*100, 2))+'%' + "(" +
                str(round((i.get_width()/len(validation_data))*100, 2))+'%' +")", fontsize=14,color='black')
        
    count += 1

fig.tight_layout()
plt.show()

#checking if acutal answer is same as persons answers

validation_data['derived_answer'] =  validation_data["answers"].apply(lambda x: getPeopleAnswer(x))

#checking if Questions has any multiple answers
print(validation_data[ validation_data['derived_answer'] != validation_data['multiple_choice_answer']])


