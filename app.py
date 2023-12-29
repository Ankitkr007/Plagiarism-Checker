from flask import Flask, render_template, request, jsonify
import pickle
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score


app = Flask(__name__)

# Load the model
# Load the model and threshold
model = pickle.load(open('model.pkl','rb'))




# Function to scrape original content from a URL
def scrape_content(url):
    import requests
from bs4 import BeautifulSoup
import re

import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
scrap_raw_text={}
def scrape_content(url):
    try:
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            

            # Identify the heading and footer elements and exclude them
            heading = soup.find('header')
            footer = soup.find('footer')
            if heading:
                heading.extract()
            if footer:
                footer.extract()

            # Extract the remaining content
            content = soup.get_text(separator=' ')
            content = re.sub('\s+', ' ', content).strip()
            scrap_raw_text['text']=content
            
            # Split the content into separate elements based on full stops and new lines
            sentences = content.split('.')

            # Remove all punctuation marks and convert to lowercase, and remove extra spaces
            cleaned_sentences = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in sentences]

            # Remove empty strings
            cleaned_sentences = list(filter(None, cleaned_sentences))

            # Filter sentences with a length of at least 15 characters
            original_content = [elem for elem in cleaned_sentences if len(elem) >= 15]
            print(original_content)
            scrap_raw_text['content']=original_content
            return original_content
        else:
            print(f"Error: Unable to access {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to access {url}")
        return None

def plag_text(text_snippet):
    # Step 1: Remove extra spaces and "\xa0"(non-breaking space) 
    
    cleaned_string = re.sub(r'\s+|\\xa0', ' ', text_snippet).lower().strip()

    # Step 2: Remove all punctuation except "..." and "." (single full stop)
    cleaned_string = re.sub(r'[^\w\s.]|(?<!\.)\.\.(?!\.)', '', cleaned_string)

    # Step 3: Split the string at occurrences of "...", ".", or any whitespace
    result_list = [elem.strip() for elem in re.split(r'\.\.\.|(?<!\.)\.(?!\.)', cleaned_string)]

    # Remove any empty elements from the list and remove the full stop "." from each element
    result_list= [elem.replace('.', '') for elem in result_list if elem and elem != ' ']
    print(result_list)
    return result_list


# Calculate cosine similarity

def calc_cosine_similarity(plag, original):
    # Join the lists of strings into single strings
    plag_text = ' '.join(plag)
    orig_text = ' '.join(original)
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the text into TF-IDF vectors
    tfidf_matrix_plag = vectorizer.fit_transform([plag_text])
    tfidf_matrix_orig = vectorizer.transform([orig_text])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix_plag, tfidf_matrix_orig)
    print(plag_text)
    return cosine_similarities.diagonal()

# Calculate basic parameters
# Calculate basic parameters
def calculate_parameters(plag_texts,original_texts):
    # Calculate the number of words in the original text
    total_words_original = sum(len(sentence.split()) for sentence in original_texts)

    # Calculate the number of characters in the original text
    total_char_original = sum(len(sentence) for sentence in original_texts)
    # Calculate the number of words in the plagiarized text
    total_words_plag = sum(len(sentence.split()) for sentence in plag_texts )

    # Calculate the number of characters in the plagiarized text
    total_char_plag = sum(len(sentence) for sentence in plag_texts)

    # Calculate the ratio of words in the plagiarized text
    ratio_word = total_words_plag / total_words_original

    # Calculate the ratio of characters in the plagiarized text
    ratio_char = total_char_plag / total_char_original

    # Calculate the number of elements (sentences) in the original text
    element_count_original = len(original_texts)

    # Calculate the number of elements (sentences) in the plagiarized text
    element_count_plag = len(plag_texts)

    # Calculate the ratio of elements (sentences) in the plagiarized text
    ratio_element = element_count_plag / element_count_original

    basic_parameters = {
        'total_words_original': total_words_original,
        'total_char_original': total_char_original,
        'total_words_plag': total_words_plag,
        'total_char_plag': total_char_plag,
        'ratio_word': ratio_word,
        'ratio_char': ratio_char,
        'element_count_original': element_count_original,
        'element_count_plag': element_count_plag,
        'ratio_element': ratio_element
    }
    return basic_parameters
# Calculate similarity score for each row
def calculate_similarity_score(plag_text, original_text):
    similarity_percent = []
    for s2 in plag_text:
        found = False  # Use a boolean flag instead of 0/1
        percent_match = 0
        for s1 in original_text:
            if s1.find(s2) >= 0:
                percent_match = int(len(s2.split(" ")) / len(s1.split(" ")) * 100)
                found = True
                break
        if not found:  # Use 'not' instead of 'found == 0'
            print(s2)
        similarity_percent.append(percent_match)
    print(similarity_percent)
    return similarity_percent
# Calculate count and percentage for similarity ranges
def calculate_similarity_ranges(similarity_list):
    count_100_similarity = sum(1 for element in similarity_list if element == 100)
    count_75_100_similarity = sum(1 for element in similarity_list if 75 <= element < 100)
    count_50_75_similarity = sum(1 for element in similarity_list if 50 <= element < 75)
    count_25_50_similarity = sum(1 for element in similarity_list if 25 <= element < 50)
    count_below_25_similarity = sum(1 for element in similarity_list if element < 25)
    
    total_elements = len(similarity_list)
    
    percentage_100_similarity = (count_100_similarity / total_elements) * 100
    percentage_75_100_similarity = (count_75_100_similarity / total_elements) * 100
    percentage_50_75_similarity = (count_50_75_similarity / total_elements) * 100
    percentage_25_50_similarity = (count_25_50_similarity / total_elements) * 100
    percentage_below_25_similarity = (count_below_25_similarity / total_elements) * 100
    
    # Return a dictionary instead of a tuple
    print({
        'count_100_similarity': count_100_similarity,
        'count_75_100_similarity': count_75_100_similarity,
        'count_50_75_similarity': count_50_75_similarity,
        'count_25_50_similarity': count_25_50_similarity,
        'count_below_25_similarity': count_below_25_similarity,
        'percentage_100_similarity': percentage_100_similarity,
        'percentage_75_100_similarity': percentage_75_100_similarity,
        'percentage_50_to_75_similarity': percentage_50_75_similarity,
        'percentage_25_to_50_similarity': percentage_25_50_similarity,
        'percentage_below_25_similarity': percentage_below_25_similarity
    })
    return {
        'count_100_similarity': count_100_similarity,
        'count_75_100_similarity': count_75_100_similarity,
        'count_50_75_similarity': count_50_75_similarity,
        'count_25_50_similarity': count_25_50_similarity,
        'count_below_25_similarity': count_below_25_similarity,
        'percentage_100_similarity': percentage_100_similarity,
        'percentage_75_100_similarity': percentage_75_100_similarity,
        'percentage_50_to_75_similarity': percentage_50_75_similarity,
        'percentage_25_to_50_similarity': percentage_25_50_similarity,
        'percentage_below_25_similarity': percentage_below_25_similarity
    }

def main(plagiarised_text, url):
    # Step 1: Scrape content from the URL
    original_doc = scrape_content(url)
    
    if original_doc is None:
        print("Error: Unable to scrape content from the provided URL.")
        return None

    # Step 2: Clean the plagiarised text
    
    plagiarised_doc = plag_text(plagiarised_text)
    
    # Step 3: Calculate Cosine Similarity
    similarity = calc_cosine_similarity(plagiarised_doc,original_doc)[0]
       #print("Cosine Similarity:", similarity)

    # Step 4: Calculate Basic Parameters
    basic_params = calculate_parameters(plagiarised_doc, original_doc)

    # Step 5: Calculate Similarity Parameters
    similarity_scores = calculate_similarity_score(plagiarised_doc, original_doc)
    similarity_params = calculate_similarity_ranges(similarity_scores)
    

    # Combine all parameters into a list
    parameters = [
        similarity,
        basic_params['total_words_plag'],
        basic_params['total_char_plag'],
        basic_params['total_words_original'],
        basic_params['total_char_original'],
        basic_params['ratio_word'],
        basic_params['ratio_char'],
        basic_params['element_count_original'],
        basic_params['element_count_plag'],
        basic_params['ratio_element'],
        similarity_params['count_100_similarity'],
        similarity_params['count_75_100_similarity'],
        similarity_params['count_50_75_similarity'],
        similarity_params['count_25_50_similarity'],
        similarity_params['count_below_25_similarity'],
        similarity_params['percentage_100_similarity'],
        similarity_params['percentage_75_100_similarity'],
        similarity_params['percentage_50_to_75_similarity'],
        similarity_params['percentage_25_to_50_similarity'],
        similarity_params['percentage_below_25_similarity']
    ]

    return parameters


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_plag():
    try:
        plagiarised_text = str(request.form.get('plagiarised_text'))
        url_link = str(request.form.get('url_link'))

        # Generate parameters using the main function
        parameters = main(plagiarised_text, url_link)

        if parameters is None:
            return "Error: Unable to generate parameters."

        # Reshape parameters to be a 2D array
        parameters = np.array(parameters).reshape(1, -1)

        # Make a prediction using the model
        threshold = 0.1

        predicted_proba = model.predict_proba(parameters)

        predicted = (predicted_proba[:, 1] >= threshold).astype('int')

        if predicted == 1:
            return "Plagiarised"
        else:
            return "Not Plagiarised"
      
    except Exception as e:
        # Handle exceptions and return an error message
        return f"An error occurred: {str(e)}"
    







if __name__ == '__main__':
    app.run(debug=True, port=5003)


