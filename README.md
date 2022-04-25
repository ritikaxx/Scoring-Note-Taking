# Scoring-Note-Taking
Scoring patient note taking by doctors using NLP.


With the advancements in Natural Language Processing, the manual job of a physician to analyze every patient’s notes thoroughly to ensure a proper diagnosis of the symptoms, complaints of the patients, and their medical history can be improved. So, we propose a methodology for National Board of Medical Examiner, which accesses the skills of writing patient’s notes for Medical Licensing Examination. The process of assessing the notes for every candidate manually is very time consuming for the trained physicians. Using NLP, the task of identifying clinical concepts in patient’s notes following the exam rubric will be done.
Using NLP models like BERT, ALBERTA, DEBERTA, ROBERTA we will be showing the result of the input given by the trained physicians to analyze the patient notes for all the candidates.


## OBJECTIVE

The objective of our proposed methodology is to make the manual task of trained physicians to analyze all the candidates notes to correctly map the features or diseases with the patients symptoms, problems and medical history using NLP models.

Statements like “quitting job” and  “no longer interested on working ” referring to the same feature/ problem have to be mapped correctly according to the exam rubrics.


Another objective is combining multiple text segments or sentences having ambiguous meanings which basically correspond to a particular feature.


We will be developing a full software solution in which the input will be in the form of a csv file uploaded by the trained physicians and the output will be the mapped feature and the particular locations of the part of the notes implying the annotations for scoring the candidates.


## RUN ON LOCALHOST

Install dependencies-

    pip install -r requirements.txt
    
Run the flask app

    python main.py
    
