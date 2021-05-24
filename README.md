# Predicting Drug Ratings using DeepLearning (DL)

DeepLearning and NLP are enabling companies to leverage publicly available data such as user reviews to gauge social sentiment towards certain products. In this project, I wanted to examine how close would a DL model be able to learn to predict drug product ratings based on customer reviews. I leveraged the dataset provided in a reasearch paper "Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning" by Grasser & Kallumadi (2018). Essentially, model inputs are customer reviews and the target output variables are the product ratings as illustred below:

![image](https://user-images.githubusercontent.com/26292532/119268319-5fc82800-bbc0-11eb-92d1-ca292066fa98.png)

## Preliminary Analysis

Since the initial ratings range was between 1-10, I examined the 2 most promising models: Support Vector Machine (SVM) and Keras Sequential neural network model. Neural network performed better and I decided to pursue it for further optimization. The key decision was to compress the ratings range into 3 categories: negative (-1), neutral (0), and positive (1) and train the model accordingly. The reason was due to significant key-word overalap for similar ratings. 

For data preprocessing, I leveraged regular expression to omit irrelevant symbols from the reviews and consquently applied Word2Vec library to tokenize reviews for the NN model. Based on a 25% test sample, NN model was able to achieve 85% accuracy.


![image](https://user-images.githubusercontent.com/26292532/119268596-b5e99b00-bbc1-11eb-998e-d61a2f5acef7.png)

## Key Learnings

This project provided me with opportunity to work with a real world dataset because the reviews were scrapped from pharamaceutical company websites and were quite messy. As a results, I developed strong data cleaning abilities using regular expression and pandas. If I were to do this project again, instead of training the deeplearning model from scratch, I would try to leverage the state-of-the-art NLP models such as Transformers Pipelines & GPT-2 to achieve even better performance. 
