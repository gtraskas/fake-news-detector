# Machine Learning Engineer Nanodegree
## Capstone Project - SageMaker Fake News Detector
George Traskas  
June 16th, 2020

## I. Definition
The term ‘fake news’ regularly hits the headlines nowadays. Whether it is to do with political events or information on the various social platforms, it seems like it is getting harder and harder to know who to trust to be a reliable source of information. This project is about understanding of what is actually fake news and propose an efficient way to detect them using the latest machine learning and natural language process techniques.

In more detail, in this project I developed a machine learning program to identify when a news source may be producing fake news. I used a corpus of labeled real and fake new articles to build a classifier that can make decisions about information based on the content from the corpus. The model focus on identifying fake news, based on multiple articles originating from a source. Eventually, the model can predict with high confidence that any future articles from that source will also be fake news. Focusing on sources widens our article misclassification tolerance, because we will have multiple data points coming from each source.

The intended application of the project is for use in applying visibility weights in social media. Using weights produced by this model, social networks can make stories, which are highly likely to be fake news less visible.

### Project Overview

The project’s desired results to build are:

- a model in which text can be provided as input and

- predict if it is fake or true.

Specifically, in my study, I investigated:

- The performance of a fast traditional machine learning model, such as Naive Bayes and used this as a Proof of Concept to move on a second analysis in AWS.

- The performance of the AWS BlazingText machine learning algorithm, which then I deployed and put it on work building a simple web page application, where a user can input some text and get the prediction if this news text is fake or true.

**Datasets**

A dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) was used as a training, validation, and test input for the algorithms that were used. The dataset features a list of articles, together with the subject of the article and its title categorised as ‘Fake’ or ‘True’.

**Acknowledgements for the dataset:**

> Ahmed H, Traore I, Saad S. “Detecting opinion spams and fake news using text classification”, Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018.

> Ahmed H, Traore I, Saad S. (2017) “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138).

### Problem Statement
I decided that it is more wise to first create a Proof of Concept (PoC) locally in my machine (even with a fraction of data) in order to demonstrate that the proposed solution has a practical potential. Then, I proceeded to an end to end solution deploying a ML model in AWS. For the implementation, I followed the next steps:

- Explored data to derive useful insights and get a better feeling of the data (EDA). I made several plots to check balance of data and word clouds to get the feeling of the most frequent words and bigrams.
- Cleaned, simplified, and prepared the dataset in a proper format:
    - Converted the entire document’s text into lower case, so that capitalisation is ignored (e.g., IndIcaTE is treated the same as Indicate).
    - Normalised numbers, i.e. replaced all numbers with the text “number”.
    - Removed non-words and punctuation, as well as trimmed all white spaces (tabs, newlines, spaces) to a single space character.
    - Tokenised, i.e. break up sequence of strings into pieces, such as words, keywords, phrases, symbols and other elements called tokens. Tokens can be individual words, phrases or even whole sentences. In the process of tokenisation, some characters like punctuation marks were discarded.
    - Stemmed words, i.e. reduced words to their stemmed form. For instance, “discount”, “discounts”, “discounted” and “discounting” will be all replaced with “discount”.
    - Applied other normalising techniques (e.g. html, link normalisation, etc.) and tested models for the best combinations.
- Implemented a bag-of-words model.
    Since we cannot work with text directly when using machine learning algorithms, I needed to convert the text to numbers. Algorithms take vectors of numbers as input, therefore I needed to convert documents to fixed-length vectors of numbers. A simple and effective model for thinking about text documents in machine learning is called the bag-of-words model, or BoW.
The model is simple in that it throws away all of the order information in the words and focuses on the occurrence of words in a document. This can be done by assigning each word a unique number. Then, any document we see can be encoded as a fixed-length vector with the length of the vocabulary of known words. The value in each position in the vector could be filled with a count or frequency of each word in the encoded document. This is the bag of words model, where we are only concerned with encoding schemes that represent what words are present or the degree to which they are present in encoded documents without any information about order. The Scikit-learn library provides 3 different schemes that we could use: CountVectorizer,
TfidfVectorizer, HashingVectorizer.
- Split the dataset to train, validation, and test sets.
- Trained a simple and fast model (Naive Bayes offered by Sklearn module) to get some first results from our data. This was done locally as a PoC before proceeding to a paid service for deployment, such as AWS.
- Evaluated performance with various metrics, i.e. accuracy score, confusion matrix and report (precision, recall, f1 scores), confusion plot.
- Prepared data for AWS SageMaker BlazingText algorithm.
- Trained the model and saved its artefacts in S3.
- Deployed the model in another instance.
- Tested with the unseen test data located in S3. Evaluated with accuracy score.
- Created a Lambda function that prepares data/text input from a REST API.
- Tested everything deploying a simple Web App html. The endpoint can be called from a simple HTML webpage. In this page, the user is able to post a text and check whether is true or fake news.

The application architecture diagram of the used AWS services is:

![alt text](https://github.com/gtraskas/fake-news-detector/blob/master/web-app-diagram.svg)

### Metrics
The dataset properly split to train, validation, and test sets before doing any transformations of the data and induce data leakage. The validation test can be used for the model tweaking in order to optimize its hyper-parameters. The dataset was clearly balanced so accuracy score only can be sufficient as an evaluation metric. However, I also used a classification report, confusion matrix and plot provided by the Sklearn framework. These offers more metrics, such as precision, recall, and f1 scores, which can depict all the type of errors of our models (Type I and Type II errors).

## II. Analysis

### Data Exploration and Visualizations
The analysis of the EDA can be read:

- either downloading the `fake-news-detector/data-exploration.html` (best experience to see the Plotly charts),
- or having a look at the `fake-news-detector/data-exploration.ipynb`

### Algorithms, Techniques, Benchmarks, and Methodologies
These sections are discussed and presented thoroughly in the following file:

- `fake-news-detector/fake-news-detector.html` or
- `fake-news-detector/fake-news-detector.ipynb`

### Data Preprocessing
Data processing is made during the Exploratory Data Analysis so as to create word clouds and other plots presenting the words frequency. In the `fake-news-detector.ipynb` I used various helper functions from the `helper.py` in order to prepare and processed data with Natural Language Techniques, which were discussed earlier.

Specifically, the training/validation data for the `BlazingText` algorithm must be processed appropriately so as to contain a training sentence per line along with the labels. Labels are words that are prefixed by the string __label__. Here is an example of a training/validation file:

```
__label__1  linux ready for prime time , intel says , despite all the linux hype , the open-source movement has yet to make a huge splash in the desktop market . that may be about to change , thanks to chipmaking giant intel corp .

__label__0  bowled by the slower one again , kolkata , november 14 the past caught up with sourav ganguly as the indian skippers return to international cricket was short lived . 
```
### Implementation, Results, and Evaluation Tests
These sections are presented thoroughly in the following file:

- `fake-news-detector/fake-news-detector.html` or
- `fake-news-detector/fake-news-detector.ipynb`

The final model can predict with high accuracy (over 98%) if a news text is fake or true. The deployed model was tested using a simple web page and insert some news text. The output files can be seen under the `fake-news-detector/data` folder. The model's accuracy was derived using an unknown test set, so the results can be trusted.


## III. Conclusion and Reflection
It seems that the model can classify with very high accuracy, precision, and recall fake and true news from a specific news source. This could be extended with more data from various sources and eventually create a viable solution for the end user as a helpful guide to detect fake news.

AWS can offer us all the sources needed to create and scale-up such a system. The dataset I used here was about 44,000 examples with about 110 MB of size. This dataset was trained in about few seconds in a simple AWS machine learning instance, so even if we use a much larger dataset with millions of data, we will get fast results. BlazingText is a very fast AWS algorithm and SageMaker provides all the functionality to build a working production solution with the minimal cost.

### Improvement
Things to improve:

- Use a lot more data from various news sources.
- Optimize hyper-parameters using the AWS services.
- Implement a higher quality web page for the end user to insert the news text.
- Export as a result except the Fake or True label, the probability of a news text to be fake or true. For example, it is more informative for the user to know that this news text was detected 65% as a Fake. This means that there are 35% probability to be true.

![alt text](https://github.com/gtraskas/fake-news-detector/blob/master/data/ml-engineer-certificate.png)