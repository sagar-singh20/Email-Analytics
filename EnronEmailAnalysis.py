#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:53:32 2019

@author: root
"""

import pandas as pd
import numpy as np
import email

#Data location: https://www.kaggle.com/wcukierski/enron-email-dataset/kernels

enronEmails = pd.read_csv("/Users/sagarsingh/Downloads/emails.csv")

emailCorpus =[]

for index,mail in enronEmails.iterrows():
    
    msgBody = email.message_from_string(mail["message"])
    
    emaildictionary = {}
    print("Mail count : ", index)
    
    for item in msgBody.items():
        emaildictionary[item[0]] = item[1]
        
    emaildictionary["Mail Body"] = msgBody.get_payload()
        
    emailCorpus.append(emaildictionary)
       
email_df = pd.DataFrame(emailCorpus)

email_df.to_csv("/Users/sagarsingh/Downloads/EnronEmailDataset.csv", index=False)