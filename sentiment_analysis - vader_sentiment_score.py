# Sentiment Analyse
# Dietmar Benndorf
# 2024


import pandas
from alive_progress import alive_bar
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main(file):

    output_file = 'modified_' + str(file)
    df_org = pandas.read_csv(file, sep=';')
    df_mod = pandas.read_csv(output_file, sep=';')
    _temp_list = []

    with alive_bar(len(df_org), force_tty=True) as bar:

        for index, row in df_org.iterrows():
            _temp_list.append(fill_sentiment_score(row))
            bar()

        df = df_mod.assign(Sentiment_Score=_temp_list)

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

    return ''


def sentiment_analysis(text):
    """
    Does a sentiment analysis of a text and returns the sentiment score (vaderSentiment).
    The score ranges from 1 (positive) to -1 (negative).
    """

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)

    return vs['compound']


if __name__ == '__main__':
    main('NatSciMaster-20240221.csv')
