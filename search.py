from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
import json
json_path = 'products.json'

def get_sentiment_score(review):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(review)
    # -1 -> 0 -> 1 = negative -> neural -> positive
    return sentiment_score['compound']

# Iterate through the products to get weighted average of each
def iterate(data):
    for product in data:
        reviews = product.get('reviews', [])
        scores = []
        for review in reviews:
            review_text = review.get('review', '')
            profile_score = review.get('profile_score', 0)  # Default to 0 if not present
            sentiment_score = get_sentiment_score(review_text)
            scores.append(sentiment_score * profile_score)

        product_dict[product['product_id']] = calculate_weighted_average(scores)
    
    product_dict = sorted(product_dict.items(), key=lambda item: item[1], reverse=True)

    simple_product_dict = []
    for i in product_dict:
        simple_product_dict.append(i[0])
    return simple_product_dict
    
    return product_dict

def calculate_weighted_average(scores):
    weights = list(range(len(scores), 0, -1))
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    weighted_average = weighted_sum / total_weight
    return weighted_average