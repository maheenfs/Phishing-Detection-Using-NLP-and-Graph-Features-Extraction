import pandas as pd
import networkx as nx
from urllib.parse import urlsplit

def create_graph_from_urls(urls):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph (each node represents a URL)
    for url in urls:
        G.add_node(url)

    # Extract domains from URLs
    domains = [urlsplit(url).netloc for url in urls]

    # Create a dictionary to store URLs by their domain
    urls_by_domain = {}
    for url, domain in zip(urls, domains):
        if domain not in urls_by_domain:
            urls_by_domain[domain] = []
        urls_by_domain[domain].append(url)

    # Add edges between URLs that share the same domain
    for domain, domain_urls in urls_by_domain.items():
        for i in range(len(domain_urls)):
            for j in range(i+1, len(domain_urls)):
                G.add_edge(domain_urls[i], domain_urls[j])

    return G

def compute_pagerank_scores(G):
    return nx.pagerank(G)

def extract_graph_features(dataset):
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)
    urls = dataset['URL'].tolist()

    G = create_graph_from_urls(urls)
    pagerank_scores = compute_pagerank_scores(G)

    # Convert the dictionary of PageRank scores into a DataFrame
    pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['URL', 'PageRank'])

    return pagerank_df


def extract_graph_features_single_url(url):
    # Create a DataFrame with a single URL
    url_data = {'URL': [url]}
    single_url_df = pd.DataFrame(url_data)
    print("Graph features before processing:")
    print(graph_features)

    # Call the extract_graph_features function with the single URL DataFrame
    graph_features = extract_graph_features(single_url_df)
    print("Graph features after processing:")
    print(processed_graph_features)
    import pandas as pd
    import networkx as nx
    from urllib.parse import urlsplit

    def create_graph_from_urls(urls):
        # Create an empty directed graph
        G = nx.DiGraph()

        # Add nodes to the graph (each node represents a URL)
        for url in urls:
            G.add_node(url)

        # Extract domains from URLs
        domains = [urlsplit(url).netloc for url in urls]

        # Create a dictionary to store URLs by their domain
        urls_by_domain = {}
        for url, domain in zip(urls, domains):
            if domain not in urls_by_domain:
                urls_by_domain[domain] = []
            urls_by_domain[domain].append(url)

        # Add edges between URLs that share the same domain
        for domain, domain_urls in urls_by_domain.items():
            for i in range(len(domain_urls)):
                for j in range(i + 1, len(domain_urls)):
                    G.add_edge(domain_urls[i], domain_urls[j])

        return G

    def compute_pagerank_scores(G):
        return nx.pagerank(G)

    def extract_graph_features(dataset):
        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset)
        urls = dataset['URL'].tolist()

        G = create_graph_from_urls(urls)
        pagerank_scores = compute_pagerank_scores(G)

        # Convert the dictionary of PageRank scores into a DataFrame
        pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['URL', 'PageRank'])

        return pagerank_df

def extract_graph_features_single_url(url):
    # Create a DataFrame with a single URL
    url_data = {'URL': [url]}
    single_url_df = pd.DataFrame(url_data)

    # Call the extract_graph_features function with the single URL DataFrame
    graph_features = extract_graph_features(single_url_df)

    print("Graph features after processing:")
    print(graph_features)

    return graph_features.iloc[0]

