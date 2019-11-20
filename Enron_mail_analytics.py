#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:54:53 2019

@author: sinjinibose
"""

import email
import pandas as pd
data = pd.read_csv("//Users//sinjinibose//Downloads//emails.csv")
data = pd.DataFrame(data)
li_attributes = ['Message-ID','Subject','To','From','Date_and_Time','X-cc','X-bcc','X-FileName']
actual_data = pd.DataFrame()
actual_data['message_body'] = data['message'].apply(lambda x: email.message_from_string(x).get_payload())##getting the email bodies and storing 

for i in li_attributes:
    actual_data[i] = data['message'].apply(lambda x: email.message_from_string(x).get(i))## getting the rest pf the attributes and storing them in the final dataset
    