# Conduct Sentiment Aanalysis on the Text Messages

## Overview

This NLP project uses deep learning (specifically, an LSTM seq2seq network with attention mechanisms) to build a chatbot capable of identifying message senders based on SMS data.  It enhances performance through rule-based or retrieval methods and features a user-friendly interface.  The project's core is sentiment analysis of the SMS data.

## Goal

In this project, we'll create [a presentation that showcases sentiment analysis](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing), leverage it for text classification using a Naive Bayes classifier to predict the sentiment of our chats, and a well-built generative chatbot system:
1. Apply  deep-learning knowledge and chatbot savvy to build an open-domain generative chatbot using a stream of sms. During this project, 
2. Use a seq2seq LSTM network built with Keras and create a chatbot that is able to handle user input and language that wasn’t part of the training data;
3. Build a user interface (website) for the chatbot;
4. Try to include [attention in the seq2seq network](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3);
5. Add in some rule-based or retrieval-based methods to improve the experience further.

## Data
- Provided  dataset `clean_nus_sms.csv` got from [Codecademy](https://www.codecademy.com/). 
- Contains information about 48598 sms senders.
- Each row contains information about a message of a particular sender with his identity, his country, the length  of his text message, and the date in which the  message was sent.
- The columns represent a set of five fields which contains user ID, length, country,  message  and date.




## Action

1. Import necessary libraries
2. Data Preprocessing
   - **NLT Data Downloads**: Ensure we have the necessary NLTK data files for tokenization and lemmatizaation using `nltk.download`.
   
    - **Tokenization**: Use `word_tokenizze` to split the text into words.

    - **Stemming**: Use `PorterStemmer` to reduce words to their root form.
   
    - **Lemmatization**: Use `WordNetLemmatizer` to reduce words to their canonical form.
  
      
    - **Rejoin Tokens**: After stemming and lemmatization, join the tokens back into a single string.
  
3. Enhance the data preprocessing by including POS (Part of Speech) tagging and filtering only specific types of tokens, such as adjectives, adverbs, and verbs.
    - **POS Tagging**: The `pos_tag` function from NLTK assigns a part of speech to each token.
    - **Filter Tokens**: Using the POS tags, we filter out only the tokens that are adjectives (JJ, JJR, JJS), adverbs (RB, RBR, RBS), and verbs (VB, VBD, VBG, VBN, VBP, VBZ).
    - **Remove Stop Words**: Commonly used words (e.g., "the", "is") are removed because they usually do not contribute significant information for most analyses.
    - **Stemming and Lemmatization**: Stem and lemmatize the filtered tokens.
    - **Rejoin Tokens**: Join the filtered, stemmed, and lemmatized tokens back into a single string.

4. Use the Keras library to create a seq2seq LSTM network with an attention mechanism.
5. Create a simple web-based user interface using Flask.
6. Create an index.html file for the web interface
7.. Integrate rule-based methods to handle specific questions or retrieval-based methods to provide accurate information from the dataset.

## Results and Conclusion
This sentiment analysis project has successfully demonstrated how machine learning models, specifically LSTM (Long Short-Term Memory) networks, can be utilized to analyze text data for sentiment. The implementation involved preprocessing text data, training the model, and deploying it through a web-based interface.

In summary, while the sentiment analysis project serves as a solid foundation for understanding and employing machine learning techniques in text analysis, there are numerous avenues for future enhancement and application. This adaptability will allow the project to evolve and remain useful as new technologies and data sources emerge.
