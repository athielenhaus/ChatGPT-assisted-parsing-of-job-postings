# Extraction and Classification of Job Posting Features via LLM

This project is inspired by the Indeed online job portal, which has a feature that extracts and prominently displays key information, such as salary range and company benefits, from listed job postings.  

<img src="images/screenshot_ext_features.png" alt="screenshot of extracted features" width="300"/>  

## Business Case

For job portals, being able to extract key features from lengthy job postings can be very useful. The prominent display of salary ranges, company benefits and work conditions can help job portal users quickly decide whether a job posting is of interest to them, thereby allowing them to save time and improving their overall experience.  

Furthermore, the job portal can use such features to create an ML model which predicts the popularity of job postings (user engagement in the form of clicks, saves, shares etc.). The portal can use this to provide constructive feedback to employers who use the portal to list their open positions.  

## Objective

For this project we focus on the second part of the business case mentioned above. The aim is to create a model or program which, as a first step, determines whether or not the job posting contains certain important job posting elements. This classification model should have a minimum 90% accuracy. The toolkit is the OpenAI GPT API in combination with prompt engineering.

I look at three important job posting elements:
- part time option (i.e. the position can be done part-time)
- salary range (ex: 50,000 - 60,000 euros, $20 per hour, etc.)
- company benefits (ex: company car, device, gym membership, insurance, etc.)

## Results

### Accuracy
Via a two-step model, I was able to successfully achieve at least 90% accuracy on all three elements. In a number of cases, the GPT model identified items that I had overlooked during manual inspection. 

* __Part-time Options:__ 96-100% accuracy. The GPT model proved in fact to be superior to the Indeed model, which incorrectly categorized a position with the following statement as part-time: "It is anticipated that the candidate will be willing to work in-person or hybrid out of our office in Barcelona, Spain full-time. However, qualified candidates wishing to telecommute full or part-time and able to work in the European Union or the UK, will be considered on a case-by-case basis." The only ambiguous classification resulted from equivocal information in the job posting, which also a human would be unable to resolve. 

* __Salary Ranges:__ 100% accuracy. The salary ranges were very easy for the two-step model to pick up. 

* __Company Benefits:__ 92% accuracy. A problem here is that it is difficult to define what exactly a company benefit is. The model classified the following as company benefit: "Young and stimulating work environment - Part-time job (from 8 to 12 hours a week) - Boost your CV: add teaching experience to your skill set." This is somewhat subjective and could even be difficult for a human to classify. The other (also debatable) misclassification as "company benefit" involved the following "Travel expenses reimbursed and accommodation provided - Seasonal position with work offered on a tour-by-tour basis - Salary range of 900.00€ - 1,500.00€ per week". These examples illustrate that, for such feature extraction / classification tasks involving natural language, it is necessary to use very precise definitions or otherwise accept a level of ambiguity. 

### Costs
The average job posting in the samples had a length of 3440 characters, which amounts to around 800 tokens. Including the prompts and the API outputs, the number of tokens was around 860. While the price of output tokens is slightly higher than that of input tokens, output tokens in most cases account for a small fraction of the tokens generated in the process. Using the gpt-3.5.-turbo model, the average cost per posting is therefore still below 0.0015 USD. For 1 million job postings, this amounts to around US$ 1230. Through chunking, NLP methods (removing punctuation, spaces, etc.), prompt adjustments and other optimization, this figure can presumably be further reduced.

### Advantages & Disadvantages
Using an LLM like GPT to address this type of problem brings several advantages:
- no model training, only prompt engineering. The iteration cycles are much shorter, so that a decent model can be created in 1-2 days.
- multilingual capabilities. The sample dataset, consisting of job postings from the Barcelona area, included a few job postings which are partially in Spanish. The results show that the model is able to correctly extract and classify elements _even if they are in a language other than the prompt, and without explicit instruction_. This is incredibly useful for markets where multiple languages are spoken.

Disadvantages of using the OpenAI GPT API include latency, rate limits and reliability. During experimentation, the process was interrupted on multiple occassions due to ServerUnavailable Errors. For an application which requires high availability and getting lots of results quickly, creating an in-house model is (for the moment) still going to be the better solution. However, the gathering of training data can be significantly supported by a model such as this one.

## Approach

To achieve the objective, the following steps were necessary:
- get sample data
- create baseline prompts to extract information
- check accuracy using sample data
- improve and iterate by adjusting prompt content as well as number of steps / prompts / API requests

### Get sample data

In a first step, I collected 20 English-language job postings from Indeed and saved them in a Google Sheets file. I visually inspected each posting and determined whether the posting contained the elements mentioned above. For each of the three elements, I created a column with boolean values ("1" if the element is present, "0" if not). I downloaded this sheet as a CSV file in order to easily upload it to Google Colab.

### Determine performance measurement

The question which the model attempts to answer is: "Does the job posting include job requirements / salary range / company benefit, yes or no?"
Since it is a classification problem, using accuracy in combination with a confusion matrix is an appropriate method to measure performance. 
Due to LLM tendency to hallucinate, we can also consider False-Positive Rate (FPR) in particular.

### Prompt Engineering

The first attempt involved creating a single, zero-shot prompt which instructed the LLM to determine the presence of all the elements with a single API request. 

Since the accuracy was poor, I iterated and gradually increased the complexity of the prompt template to include multiple examples:

<pre>
  sample_prompt = f'''Determine if the job description below contains examples of the following elements:
  1) specific job requirements (examples: "python", "communication skills", "university degree", "project management experience", etc.)
  2) specific company benefits (examples: "company car", "gym membership", "private insurance", etc.)
  3) salary or salary range (examples: "50000 USD", "55000 - 75000 EUR", "150 £ per hour", etc.)

  desired output format: provide a python dictionary with binary values: "0" if examples are not present, "1" if examples are present.

  job description 1: "The company offers an attractive starting salary and benefits dependent on experience."
  output 1: {{"job_reqs": 0, "company_benefits": 0, "salary_range": 0}}

  job description 2: "As a sales staff member at Markus LLC you can earn $15-25 per hour."
  output 2: {{"job_reqs": 0, "company_benefits": 0, "salary_range": 1}}

  job description 3: "At least two years of professional experience in machine learning, proficiency in Python and popular machine learning frameworks such as Scikit. We offer a dog-friendly office and gym membership"
  output 3: {{"job_reqs": 1, "company_benefits": 1, "salary_range": 0}}

  job description 4: {job_desc}
  output 4:
  '''
</pre>

However, the model continued to falsely indicate the presence of "salary range" and "company benefits" if the posting contained a statement such as "The position includes an attractive salary range and benefits". I therefore decided that a multi-step process focusing on each element individually might be easier, without significantly increasing costs, as it would allow for simpler prompts.

Second approach:

For each of the three elements:
- Step 1: extract features
- Step 2: analyze features

<pre>
  # Extraction Prompt
  sample_prompt = f'''Does the job description located between the triple hashtags below mention any {element}s?
  If yes, list maximum 3, using maximum 8 words for each. If no, write "no element detected". 
  Desired output format is a list:
  - {element} 1
  - {element} 2
  - {element} etc.
  
  ###
  {job_desc}
  ###
  '''
  
  # Analysis prompt (takes as an argument the API response to previous prompt)
  final_prompt = f'''Given the following phrases, please determine if they include specific {element}s commonly found in a job postings.
    Respond with "1" if yes, "0" if no.
  
    ###
    {phrases}
    ###
    '''
</pre>

This approach improved performance.


