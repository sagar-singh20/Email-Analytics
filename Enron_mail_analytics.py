#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:54:53 2019

@author: sinjinibose
"""

import email
import pandas as pd
from datetime import datetime
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize


data = pd.read_csv("//Users//sinjinibose//Downloads//emails.csv")
data = pd.DataFrame(data)
li_attributes = ['Message-ID','Subject','To','From','Date','X-cc','X-bcc','X-Folder']
actual_data = pd.DataFrame()
actual_data['message_body'] = data['message'].apply(lambda x: email.message_from_string(x).get_payload())##getting the email bodies and storing 

for i in li_attributes:
    actual_data[i] = data['message'].apply(lambda x: email.message_from_string(x).get(i))## getting the rest pf the attributes and storing them in the final dataset
copy = actual_data.copy
##removing \n and \t from the message body
temp = actual_data['message_body'].apply(lambda x : x.replace("\n",""))
actual_data['message_body'] = temp.apply(lambda x : x.replace("\t",""))

##formatting the dates(don't think we need to keep the time or timezone and the days)
only_date_and_time = (actual_data['Date']).apply(lambda x : x.split(",")[1].split("-")[0].strip())
formatted = only_date_and_time.apply(lambda x : datetime.strptime(x,"%d %b %Y %H:%M:%S"))
actual_data['Date'] = formatted.apply(lambda x : x.strftime("%m/%d/%Y"))
actual_data = actual_data.dropna()
##formatting the folder names into just inbox, sent items, etc , like removing the employee names
actual_data['X-Folder'] = actual_data['X-Folder'].apply(lambda x : x.split("\\")[-1])


#filtering data on duplicate messages
duplicate_data = actual_data[actual_data['message_body'].duplicated()==True]##257263 duplicate mails

#getting the forwarded mails (duplicate mails with FW: tag in subject, because they will be of importance)
Forwarded_mails = duplicate_data[duplicate_data['Subject'].apply(lambda x : x.__contains__("FW:")) == True] ## 11250  duplicate mails that have been forwarded
indices = Forwarded_mails.index.tolist()##indexes of the forwarded mails

##removing the subset of forwarded mails from the duplicate mails (these are the actual rows to be dropped)
data_to_be_dropped = duplicate_data.drop(labels = indices) ##257263-11250 = 246013
indices_ = data_to_be_dropped.index.tolist()

##dropping the actual duplicates from the original data 
revised_data = actual_data.drop(labels=indices_).reindex()

unique_folder_names = revised_data['X-Folder'].unique() ## 1228 unique folders

#filtering the data based on sender and receiver mappings


unique_senders = revised_data['From'].unique() 
unique_receivers = revised_data["To"].unique() 
outside_receivers = [i for i in unique_receivers if i.__contains__("enron.com") == False] ##10699 0utsiders
outside_senders = [i for i in unique_senders if i.__contains__("enron.com") == False] ## 13222 Outsiders

##checking if both sender and receiver is an outsider, so that we can eliminate such cases
##in cases where there is a group of receivers, even if there's one receiver from enron,
##then we will have to consider such cases.

rows_to_be_dropped = []
for i in outside_senders:
    temp_list = revised_data[revised_data['From'] ==i]['To'].apply(lambda x : x.__contains__('enron.com')).tolist()
    if(temp_list.count(True) == 0):
        print( i)
        rows_to_be_dropped.append(revised_data[revised_data['From']==i].index.tolist())
rows_to_be_dropped_ = [i for j in rows_to_be_dropped for i in j]        
##dropping the duplicate rows from the revised data
      

##final dataset on which we will begin our analysis  
revised_data = revised_data.drop(labels=rows_to_be_dropped_)
