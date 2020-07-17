import pandas as pd


def clean_ratings(ratings):
    min_user_books = 90

    filtered_books = ratings['book_id'].value_counts() > min_user_books
    filtered_books = filtered_books[filtered_books].index.tolist()

    filtered_users = ratings['user_id'].value_counts() > min_user_books
    filtered_users = filtered_users[filtered_users].index.tolist()

    filtered_ratings = ratings[(ratings['book_id'].isin(filtered_books)) &
                               (ratings['user_id'].isin(filtered_users))]

    filtered_ratings.to_csv("../data/processed/filtered_ratings.csv")
    print('The original dataframe shape:', ratings.shape)
    print("The new dataframe shape:", filtered_ratings.shape)


def clean_books(books):
    books = books.drop(['id','best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13',
                'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
                'image_url', 'small_image_url', 'work_ratings_count',
                'work_text_reviews_count'], axis=1)
    values = {'authors': 'NA', 'original_title': 'NA', 'language_code': 'NA'}
    books.fillna(value=values)

    # Drop missing values
    books_filtered = books.dropna(how='any')

    # Remove rows with ASCII characters
    books_filtered['language_code'].str.replace('eng-.*', 'eng')
    books_filtered = books_filtered[~books.original_title.str.contains(r'[^\x00-\x7F]', na=False)]

    books_filtered.to_csv("../data/processed/filtered_books.csv")

    print('The original dataframe shape:', books.shape)
    print("The new dataframe shape:", books_filtered.shape)


def main():
    ratings = pd.read_csv("../data/raw/ratings.csv")
    clean_ratings(ratings)

    books = pd.read_csv("../data/raw/books.csv")
    clean_books(books)


if __name__ == '__main__':
    main()