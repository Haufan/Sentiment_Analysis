# Sentiment Analyse
# Dietmar Benndorf
# 2024


import pandas
from alive_progress import alive_bar
from transformers import pipeline


def main(file):

    output_file = 'modified_' + str(file)
    df = pandas.read_csv(file, sep=';')

    with alive_bar(len(df), force_tty=True) as bar:

        for index, row in df.iterrows():
            df.at[index, 'Stance-Justification'] = fill_sentiment_score(row)
            bar()

        df.to_csv(output_file, index=False, sep=';')

    bar()


def fill_sentiment_score(row):
    """
    Checks if row "Stance-Justification" is empty.
    If so it does a sentiment analysis of the text in the same row and returns the sentiment score.
    Else it returns the original entry.
    """

    if pandas.isna(row['Stance-Justification']):
        return sentiment_analysis(row['Text'])

    return row['Stance-Justification']


def sentiment_analysis(text):
    """
    Does a sentiment analysis of a text and returns the confidence score.
    If the sentiment is negative, the score is returned as a negative score.
    """

    sent_analysis = pipeline("sentiment-analysis",
                             model="siebert/sentiment-roberta-large-english")
    result = sent_analysis(text)[0]

    if result['label'] == 'NEGATIVE':
        result = -abs(result['score'])
    else:
        result = result['score']

    return result


if __name__ == '__main__':
    main('NatSciMaster-20240221.csv')
