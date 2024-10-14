#Importing all required libraries
import pandas as pd
from newspaper import Article, ArticleException
from requests.exceptions import RequestException
from urllib.parse import quote
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease, gunning_fog, syllable_count, lexicon_count, text_standard

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

#using article library to extract article text and title
def extract_article_text(url):
    try:
        encoded_url = quote(url, safe=':/')
        article = Article(encoded_url)
        article.download()
        article.parse()
        return article.title, article.text
    except ArticleException:
        print(f"Failed : {url} (ArticleException)")
        return "", ""
    except RequestException as e:
        if '404 Client Error' in str(e):
            print(f"URL not found: {url}")
            return "", ""
        else:
            print(f"Failed: {url}")
            return "", ""

def load_stopwords(file_path, encoding='utf-8'):
    stopwords = set()
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                # Split on the first occurrence of '|' and strip any leading/trailing whitespace
                stopword = line.split('|')[0].strip()
                stopwords.add(stopword)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin1') as file:
                for line in file:
                    stopword = line.split('|')[0].strip()
                    stopwords.add(stopword)
        except Exception as e:
            print(f"Failed to load stopwords from '{file_path}': {e}")
    except Exception as e:
        print(f"Failed to load stopwords from '{file_path}': {e}")
    return stopwords

def removestopwords(text, stopwords):
    """Remove stopwords from the provided text."""
    words = text.split()
    filtered_text = ' '.join([word for word in words if word.lower() not in stopwords])
    return filtered_text

def count_words_in_text(text, words_list):
    """Count occurrences of words from a list in the provided text."""
    words = text.split()
    count = sum(1 for word in words if word.lower() in words_list)
    return count

def calculate_polarity_score(pos_score, neg_score):
    """Calculate polarity score."""
    try:
        polarity_score = (pos_score - neg_score) / (pos_score + neg_score + 0.000001)
    except ZeroDivisionError:
        polarity_score = 0.0
    return polarity_score

def calculate_subjectivity_score(pos_score, neg_score, total_words):
    """Calculate subjectivity score."""
    try:
        subjectivity_score = (pos_score + neg_score) / (total_words + 0.000001)
    except ZeroDivisionError:
        subjectivity_score = 0.0
    return subjectivity_score

def calculate_readability_metrics(text):
    """Calculate readability metrics."""
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    total_words = lexicon_count(text, removepunct=True)
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0.0
    
    # Calculate percentage of complex words
    words = word_tokenize(text)
    complex_word_count = 0
    for word in words:
        if syllable_count(word) > 2:
            complex_word_count += 1
    percentage_complex_words = (complex_word_count / total_words) * 100 if total_words > 0 else 0.0
    
    # Calculate Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    # Calculate Average Word Length
    sum_word_lengths = sum(len(word) for word in words)
    avg_word_length = sum_word_lengths / total_words if total_words > 0 else 0.0
    
    # Calculate Personal Pronouns count
    personal_pronouns_count = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text, flags=re.IGNORECASE))
    
    return {
        'Average Sentence Length': avg_sentence_length,
        'Percentage of Complex Words': percentage_complex_words,
        'Fog Index': fog_index,
        'Average Word Length': avg_word_length,
        'Personal Pronouns Count': personal_pronouns_count
    }

# Load positive words from positive-words.txt
positive_words_file = "positive-words.txt"
if os.path.exists(positive_words_file):
    try:
        with open(positive_words_file, 'r', encoding='utf-8') as file:
            positive_words = [line.strip().lower() for line in file]
    except UnicodeDecodeError:
        print(f"Error: Encoding issue with '{positive_words_file}'. Trying 'latin1'...")
        try:
            with open(positive_words_file, 'r', encoding='latin1') as file:
                positive_words = [line.strip().lower() for line in file]
        except Exception as e:
            print(f"Failed to load positive words from '{positive_words_file}': {e}")
            positive_words = []
else:
    print(f"Positive words file not found: {positive_words_file}")
    positive_words = []

# Load negative words from negative-words.txt
negative_words_file = "negative-words.txt"
if os.path.exists(negative_words_file):
    try:
        with open(negative_words_file, 'r', encoding='utf-8') as file:
            negative_words = [line.strip().lower() for line in file]
    except UnicodeDecodeError:
        print(f"Error: Encoding issue with '{negative_words_file}'. Trying 'latin1'...")
        try:
            with open(negative_words_file, 'r', encoding='latin1') as file:
                negative_words = [line.strip().lower() for line in file]
        except Exception as e:
            print(f"Failed to load negative words from '{negative_words_file}': {e}")
            negative_words = []
else:
    print(f"Negative words file not found: {negative_words_file}")
    negative_words = []

# List of custom stopwords files
stopwords_files = [
    "StopWords_Names.txt",
    "StopWords_Geographic.txt",
    "StopWords_GenericLong.txt",
    "StopWords_Generic.txt",
    "StopWords_DatesandNumbers.txt",
    "StopWords_Currencies.txt",
    "StopWords_Auditor.txt"
]

# Load and combine all stopwords
all_stopwords = set()
for stopwords_file in stopwords_files:
    if os.path.exists(stopwords_file):
        try:
            stopwords = load_stopwords(stopwords_file, encoding='utf-8')
        except UnicodeDecodeError:
            stopwords = load_stopwords(stopwords_file, encoding='latin1')
        all_stopwords.update(stopwords)
    else:
        print(f"File not found: {stopwords_file}")

# Load URLs from the Excel file
try:
    df = pd.read_excel("Output Data Structure.xlsx")
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}")
    try:
        df = pd.read_excel("Output Data Structure.xlsx", encoding='latin1')
    except Exception as e:
        print(f"Failed to read the Excel file: {e}")

# Initialize lists to store calculated metrics
positive_scores = []
negative_scores = []
polarity_scores = []
subjectivity_scores = []
average_sentence_lengths = []
percentage_complex_words_list = []
fog_indexes = []
average_word_lengths = []
personal_pronouns_counts = []

# Iterate over each URL in the DataFrame and calculate scores/metrics
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    title, article_text = extract_article_text(url)
    if title and article_text:
        # Remove stopwords from the article text
        filtered_text = removestopwords(article_text, all_stopwords)
        
        # Count positive words in the filtered text
        positive_score = count_words_in_text(filtered_text, positive_words)
        positive_scores.append(positive_score)
        
        # Count negative words in the filtered text
        negative_score = count_words_in_text(filtered_text, negative_words)
        negative_scores.append(negative_score)
        
        # Calculate polarity score
        polarity_score = calculate_polarity_score(positive_score, negative_score)
        polarity_scores.append(polarity_score)
        
        # Calculate subjectivity score
        total_words = lexicon_count(filtered_text, removepunct=True)
        subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, total_words)
        subjectivity_scores.append(subjectivity_score)
        
        # Calculate readability metrics
        metrics = calculate_readability_metrics(article_text)
        average_sentence_lengths.append(metrics['Average Sentence Length'])
        percentage_complex_words_list.append(metrics['Percentage of Complex Words'])
        fog_indexes.append(metrics['Fog Index'])
        average_word_lengths.append(metrics['Average Word Length'])
        personal_pronouns_counts.append(metrics['Personal Pronouns Count'])
        
        print(f"Processed URL ID: {url_id}, Title: {title}")
        print(f"Positive Score: {positive_score}, Negative Score: {negative_score}")
        print(f"Polarity Score: {polarity_score}, Subjectivity Score: {subjectivity_score}")
        print(f"Readability Metrics: {metrics}\n")
    else:
        positive_scores.append(0)
        negative_scores.append(0)
        polarity_scores.append(0.0)
        subjectivity_scores.append(0.0)
        average_sentence_lengths.append(0.0)
        percentage_complex_words_list.append(0.0)
        fog_indexes.append(0.0)
        average_word_lengths.append(0.0)
        personal_pronouns_counts.append(0)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'URL_ID': df['URL_ID'],
    'URL': df['URL'],
    'POSITIVE SCORE': positive_scores,
    'NEGATIVE SCORE': negative_scores,
    'POLARITY SCORE': polarity_scores,
    'SUBJECTIVITY SCORE': subjectivity_scores,
    'AVG SENTENCE LENGTH': average_sentence_lengths,
    'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words_list,
    'FOG INDEX': fog_indexes,
    'AVG WORD LENGTH': average_word_lengths,
    'PERSONAL PRONOUNS COUNT': personal_pronouns_counts
})

# Save the DataFrame to a CSV file
results_df.to_csv('output.csv', index=False)
print("Results saved to 'output.csv'")
