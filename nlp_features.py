import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import pickle
import pandas as pd

# function to cache downloaded web pages
def cache_web_page(url, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{hash(url)}.pickle")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        try:
            response = requests.get(url, timeout=10)
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
        except requests.exceptions.RequestException:
            response = None

    return response


def extract_title_and_meta_description(url):
    try:
        response = cache_web_page(url)
        if response is None or not response.content:
            return '', ''

        decoded_content = response.content.decode(errors='replace')
        if not decoded_content.strip().startswith('<'):
            return '', ''

        soup = BeautifulSoup(decoded_content, 'lxml')
        title = soup.title.string.strip() if soup.title else ''
        meta_description = ''
        for tag in soup.find_all('meta'):
            if 'name' in tag.attrs and tag.attrs['name'].lower() == 'description':
                meta_description = tag.attrs['content']
                break
        return title, meta_description
    except:
        return '', ''


# Add batch_size parameter to the parallel extraction function
def extract_titles_and_meta_descriptions_parallel(urls, max_workers=10, batch_size=1000):
    results = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(urls) // batch_size + 1}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(extract_title_and_meta_description, batch_urls))
        results.extend(batch_results)
    return results

def extract_nlp_features(urls, max_features=100, max_workers=10, batch_size=1000):
    print("Extracting titles and meta descriptions...")
    titles_and_meta_descriptions = extract_titles_and_meta_descriptions_parallel(urls, max_workers=max_workers, batch_size=batch_size)

    print("Combining texts...")
    combined_texts = [
        f"{title} {meta_description}"
        for title, meta_description in titles_and_meta_descriptions
    ]

    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    nlp_features = vectorizer.fit_transform(combined_texts)

    # Convert the csr_matrix to a DataFrame
    print("Converting to DataFrame...")
    nlp_features_df = pd.DataFrame(nlp_features.toarray(), columns=vectorizer.get_feature_names_out())

    return nlp_features_df