# Recommender-System

Ratings.dat contains 1,000,209 anonymous ratings of approximately 3,900 movies 
made by 6,040 MovieLens users who joined MovieLens in 2000.

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

"most_similar.dat" is the text file that contains top 100 most similar movies for every movie (after running on RecommenderSystem.py).

CITATION
================================================================================

To acknowledge use of the dataset in publications, please cite the following
paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872


ACKNOWLEDGEMENTS
================================================================================

Thanks to Shyong Lam and Jon Herlocker for cleaning up and generating the data
set.

File Descriptions 
================================================================================

- RecommenderSystem.py contains the implementation of Collabarative Filetering with baseline algorithm.
- SVDandCur.py          contains the implementation of SVD and CUR implementation 
- ratings.data  is the dataset 


