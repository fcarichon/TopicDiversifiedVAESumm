# Amazon Reviews Dataset (2018) with 1,000 reviews per category
## Overview
The [Amazon reviews dataset](https://jmcauley.ucsd.edu/data/amazon/) dataset (McAuley et al.,  2015) is contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2018. It includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).  For the task of text summarization, we will only use product reviews subset having the following information: `reviewerID`, `asin` (product ID), `reviewerName`, `helpful` (helpfulness score of the review), `overall` (rating assigned by the reviewer), `summary` (a summary of the review), `unixReviewTime`, `reviewTime`, `vote` (helpfulness votes), `verified`, and `style`.

The full dataset can be accessed [here](https://jmcauley.ucsd.edu/data/amazon/) and split into 29 JSON files--each file containing the reviews of products for 29 categories. The number of reviews per category varies from 171,000 reviews to 51 million. Considering the task of topic-based multi-document text summarization, we aim at having around 1,000 reviews for each product category and an average of 10 reviews per product.

After a series of filtering and processing, the final dataset constructed from the original Amazon dataset is composed of the following 5 attributes:
* **prod_id**: ID of the product, e.g. 0000013714
* **rating**: rating of the product (0-5 stars)
* **review**: text of the review
* **review_len**: number of words contained in the review's text
* **review_id**: an ID assigned to each review of each product
* **category**: category oaf the product

## CSV Data Files in Directory
* **20210629_FC_test_8r.csv**: sample test dtaset having 8 reviews and 3 summaries per product
* **20210629_FC_val_8r.csv**: sample validation dataset having 8 reviews and 3 summaries per product
* **20210629_FC_test_8r.csv**: sample train dataset having 8 reviews and 3 summaries per product
* **amazon_review_1000.csv**: raw dataset extracted from the 29 categories where each category has around 1,000 reviews
* **amazon_reviews_2018.csv**: dataset extracted from the 29 categories where each category has around 1,000 reviews and columns have been renamed
* **amazon_reviews_2018_processed.csv**: final preprocessed dataset

## Dataset Construction Process
An quick analysis of a subset of the data (7 categories out of 29) showed that around 97% of all reviews have a length of between 1 and 200 words. Moreover, the reviews having less than 8 words appeared to be quite generic, repetitive, and overall not presenting much quality information.m Because of this, we only considered the reviews having a length of 8 to 200 words. Our analysis showed that, after filtering out reviews based on length, only around 19% of products have at least 5 reviews and products have on average 8 reviews. As our goal is to have on average 10 reviews per product and 1,000 reviews per category, 19% is still a good amount--considering that 19% of reviews of the smallest product category (Gift Cards) would be around 32,500 reviews. We can, therefore, still achieve our goal of 1,000 reviews per category and 5-15 reviews per product. With this information in hand, we devised a strategy to construct a dataset suited for our tasks.

The construction of the final dataset starts with downloading all 29 JSON files from the web. We then use the Python program `build_dataset.py` to combine, filter, and preprocess the data downloaded.

### build_dataset.py
This program is made of two main components. The first step is to load the product category names from a text file. The list of categories should match the name of the JSON files downloaded (without any commas). This also allows us to specify which category to consider. Next, for each category, we load the corresponding JSON file and read it one line at a time to construct a Pandas Dataframe. Given that some files can have a size of several gigabytes, we stop reading the file after a fixed number of lines (1.5 millions by default). Once the Dataframe ready, we proceed with the filtering stage.

We start by removing every review that do not contain a single alphabet letter. Next, we create a new column `review_len` which holds the number of words contained in each review. This information will be useful when filtering reviews based on length. We also add a `category` column. We then remove from the dataset all reviews having less than 8 words and more than 200 words. Similarily, we ignore products having less than 5 reviews. Once the data is filtered, we make sure to only retain at most 15 reviews per product. At last, we append the resulting Dataframe to the final Dataframe. We load the next JSON file and repeat the steps for all the other categories. 

At the end, we are left with a dataset composed of around 29,000 reviews where each product category has on average 1,000 reviews and each product has between 8 and 15 reviews.

### preprocess.py
We provide a module for text preprocessing. The module requires two JSON files: `english_contractions.json`-- which maps contractions to their expanded form (e.g. can't --> cannot)--and `eng_abbrv.json`--mapping common english abbreviation to their expanded form (e.g. IDK --> I do not know). The preprocessing of texts is done through the class `TextPreprocessing` as follows:
* remove any markup tags (HTML, XML)
* remove URLs and hrefs
* remove emojis such as `: )`, `;-)`, `XD`, etc. as well as hexadecimal characters
* normalize punctuations by removing excessive punctuation. e.g. `!!!!!!` --> `!`
* substitute numerical values to their type. e.g. `$50` --> `some amount of money`
* remove special characters (e.g. `#`, `@`, `$`, etc.)
* expand contractions using the `english_contraction` dictionary
* expand abbreviations using the `eng_abbrv` dictionary
* correct spelling (optional): here, we reduce exageration in certain words such as `looooooooove` --> `love`
* set the review text to lowercase
* normalize whitespaces by replacing every occurence of one or many whitespaces (`\t`, `\n`, `\r`, etc) to a single space
* remove punctuations (optional)
* remove stopwords (optional)
* We end with another step of whitespace normalization to remove excessive spaces

At this satge, the data is clean and ready to be fed to a Dataloader.