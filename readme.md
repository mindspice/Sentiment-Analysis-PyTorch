Capstone project for school, uses NLP via a CNN-LSTM model to analyze string of text for sentiment assigning a
float value of 0.0 - 1.0 (Neg-Pos). Was trained using sentiment140 with binary labeling and implemented as to learn
to predict on a continuous scale.

The sentiment model for predictions needs downloaded from 
```https://f004.backblazeb2.com/file/school-bucket/sentiment_model.pt``` as it is too large to host on github.

A CLI interface is provided to make predictions on single string of text, file of multiple sentiment strings and perform
evaluation of the model against test sets. Several graphing functions are also included.

Model performs with an accuracy of 87.55 against the test set from the sentiment140 dataset, and with 90% accuracy when
evaluated against a set of 60 GPT4 generated sentiments. 