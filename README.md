# SMM4H 2025 Task 1

**Binary Classification of Social Media Posts containing Adverse Drug Events in German, French, Russian and English**

This repository contains the  evaluation script and starter-code for the SMM4H 2025 Task 1 challenge.

Website: https://healthlanguageprocessing.org/smm4h-2025/

### Overview 

Adverse Drug Events (ADEs) are negative medical side effects associated with a drug. Extracting ADE mentions from user-generated text has gained significant attention in research, as it can help detect crowd signals from online discussions. Leveraging multilingual methods to analyze ADE reports across languages and borders further enhances this effort.

For this shared task, we provide messages from patient forums, each labeled according to the presence of an ADE. A message with a positive label (1) contains at least one mention of an ADE, while a message with a negative label (0) does not.


### Task

This is a binary classification task. Given a social media post, participants are supposed to develop a system that predicts whether the post contains a mention of an Adverse Drug Event (ADE). The system should output either 1 (positive, ADE mentioned) or 0 (negative, no ADE mentioned).


### Data

The dataset consists of user-generated social media messages, where mentions of medications and medical symptoms can be highly variable and sometimes ambiguous. Additionally, the dataset is relatively small, with fewer than 2,000 documents per language. The labels are also highly imbalanced, with the positive class (posts mentioning an ADE) making up only about 1% of the data.
