# Extraction and Classification of Job Posting Features via LLM

This project is inspired by the Indeed online job portal, which has a feature that extracts and prominently displays key information, such as salary range and company benefits, from listed job postings.  

<img src="images/screenshot_ext_features.png" alt="screenshot of extracted features" width="300"/>  

## Business Case

For job portals, being able to extract key features from lengthy job postings can be very useful. The prominent display of salary ranges, company benefits and work conditions can help job portal users quickly decide whether a job posting is of interest to them, thereby allowing them to save time and improving their overall experience.  

Furthermore, the job portal can use such features to create an ML model which predicts the popularity of job postings (user engagement in the form of clicks, saves, shares etc.) based on their features. The portal can use this to provide constructive feedback to employers who use the portal to list their open positions.  

## Objective

The aim is to create a Python program which extracts features from job postings and determines whether these can be considered "key" features. The model should have a 90% accuracy. This is achieved by using the OpenAI API in combination with prompt engineering.

I look at three important job posting elements:
- candidate requirements (ex: university degree, Python skills, project management experience, etc.)
- salary range (ex: 50,000 - 60,000 euros, $20 per hour, etc.)
- company benefits (ex: company car, device, gym membership, insurance, etc.)

# Approach

To achieve the objective, the following steps were necessary:
- get sample data
- create baseline prompts to extract information
- check accuracy using sample data
- improve and iterate by adjusting prompt content as well as number of steps / prompts / API requests

### Get sample data

In a first step, I collected 20 English-language job postings from Indeed and saved them in a Google Sheets file. I visually inspected each posting and determined whether the posting contained the elements mentioned above. For each element, I created a column with boolean values ("1" if the element is present, "0" if not). I downloaded this sheet as a CSV file in order to easily upload it to Google Colab.

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

However, the model continued to falsely indicate the presence of "salary range" and "company benefits" if the posting contained a statement such as "The position includes an attractive salary range and benefits". I therefore decided that a multi-step process might be easier, without significantly increasing costs.

Second approach:
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

