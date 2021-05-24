# Drug-Reviews-Classification

Deeplearning and NLP are enabling companies to leverage publicly available user reviews to gauge social sentiment towards certain products. In this project, I wanted to examine how close would a state-of-the-art DL model be able to learn to predict drug product ratings based on customer reviews. I leveraged the dataset provided in a reasearch paper "" by Grasser & Kallamudi (2018). Essentially, the input is the customer review and the target output variable is the product rating as illustred below:

![image](https://user-images.githubusercontent.com/26292532/119268319-5fc82800-bbc0-11eb-92d1-ca292066fa98.png)

## Preliminary Analysis
Since the rating initially came between 1-10, I tried to inbluc

Here you can find a Keras model which predicts 3 categories of ratings using drug reviews. The model accuracy is 85%.



Initially, we investigated an SVM and Keras model, from which we concluded that Keras model provides better accuracy and therefore selected it for further optimization. Additionally, we performed a simple feature engineering analysis to determine the correlation between drug name, condition, and useful count features with the drug rating.

![image](https://user-images.githubusercontent.com/26292532/119268596-b5e99b00-bbc1-11eb-998e-d61a2f5acef7.png)
