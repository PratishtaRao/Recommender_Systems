import pandas as pd
import matplotlib.pyplot as plt


def eda_plots(ratings_df):
    rating_data = ratings_df['rating'].value_counts().sort_index(ascending=False)

    # Create bar plot for ratings
    plt.bar(rating_data.index.values, rating_data.values)
    plt.title("Distribution of ratings for {} books".format(ratings_df.shape[0]))
    plt.xlabel("Rating")
    plt.ylabel("Count")

    # Add percentages to each bar
    percent = []
    [percent.append('{:.1f}%'.format(val)) for val in (rating_data.values /ratings_df.shape[0] * 100)]
    for x_axis, y_axis, text in zip(rating_data.index.values, rating_data.values, percent):
        plt.text(x_axis, y_axis, str(text))
    plt.savefig("../graphs/ratings.jpg")


def main():
    book_df = pd.read_csv("../data/raw/books.csv")
    ratings_df = pd.read_csv("../data/raw/ratings.csv")
    eda_plots(ratings_df)
    print(ratings_df.shape)


if __name__ == '__main__':
    main()