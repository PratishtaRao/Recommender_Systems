import pandas as pd
import matplotlib.pyplot as plt


def rating_distribution_plot(ratings_df):
    rating_data = ratings_df['rating'].value_counts().sort_index(ascending=False)

    # Create bar plot for ratings
    plt.bar(rating_data.index.values, rating_data.values)
    plt.title("Distribution of ratings for 10000 books")
    plt.xlabel("Rating")
    plt.ylabel("Count")

    # Add percentages to each bar
    percent = []
    [percent.append('{:.1f}%'.format(val)) for val in (rating_data.values /ratings_df.shape[0] * 100)]
    for x_axis, y_axis, text in zip(rating_data.index.values, rating_data.values, percent):
        plt.text(x_axis, y_axis, str(text))
    # plt.savefig("../graphs/ratings.jpg")
    plt.clf()

    ratings_user = ratings_df.groupby('user_id')['rating'].count().clip(upper=100)
    plt.hist(ratings_user, rwidth=0.75)
    plt.xlabel("Ratings per user")
    plt.ylabel("Count")
    plt.title("Distribution of Number of Ratings Per User (Clipped at 100)")
    plt.savefig("../graphs/ratings_per_user.jpg")
    plt.clf()


def top_10_books(books_df):
    top_rated = books_df.sort_values('average_rating', ascending=False)
    top_10 = top_rated.head(10)
    print("Top rated books are:")
    print(top_10['title'])


def top_genres(tag_df):
    genre_df = tag_df.groupby("tag_name").count()
    genre_df = genre_df.sort_values(by='count', ascending=False)
    genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics",
              "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction",
              "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",
              "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",
              "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction",
              "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]
    for genre in range(len(genres)):
        genres[genre] = genres[genre].lower()
    new_tags = genre_df[genre_df.index.isin(genres)]

    plt.barh(new_tags.index, new_tags['count'])
    plt.xlabel("Count")
    plt.ylabel("Genres")
    plt.title("Popular genres for books")
    plt.show()
    # plt.savefig("../graphs/top_genres.jpg")
    plt.clf()


def main():
    # Read books data into a dataframe
    book_df = pd.read_csv("../data/raw/books.csv")
    ratings_df = pd.read_csv("../data/raw/ratings.csv")

    # Function call for plotting ratings distribution
    rating_distribution_plot(ratings_df)

    # Function call for top 10 highest rated books
    top_10_books(book_df)

    # Read book tags data into a dataframe
    book_tags_df = pd.read_csv("../data/raw/book_tags.csv")
    tags_df = pd.read_csv("../data/raw/tags.csv")

    # Join two different tags dataframe
    combined_tag_df = pd.merge(book_tags_df, tags_df, on="tag_id", how="inner")

    # Function call for plotting popular genres
    top_genres(combined_tag_df)


    #TODO: Popular language, popular author


if __name__ == '__main__':
    main()