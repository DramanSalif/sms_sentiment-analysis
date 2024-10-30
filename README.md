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
This sentiment analysis project has successfully demonstrated how machine learning models, specifically LSTM (Long Short-Term Memory) networks, can be utilized to analyze text data for sentiment. The implementation involved preprocessing text data, training the model, and deploying it through a web-based interface. Here are some findings:

1. **Distribution of sentiments in messages**:
   - The [bar chart](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) shows that the majority of messages are classified as Neutral, with a count exceeding 30,000. 
   - Positive messages are the next most common, with a count slightly above 10,000.
   - Negative messages are the least common, with a count that appears to be a little above 5,000.
     
2. **In [the word cloud](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) that encompasses words with a positive connotation**:
   - Prominent words like "good," "love," "nice," and "okay" immediately stand out due to their size and suggest affirmative and pleasant sentiments. These are surrounded by a multitude of other words:
   - Other positive words like "enjoy," "hahaha," "fine," and "happi" (which seems to be an informal or misspelled variant of "happy") contribute to the overall cheerful theme.
   - Verbs like "meet," "come," "stay," and "take" indicate actions but in a context that is likely positive, based on their inclusion in this cloud.
   - There is a playful element with words like "hahaha" suggesting laughter and jovial conversation.

3. ** In the [neutral messages cloud](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing)**:
   - The words are in various shades of green, blue, and purple, with different sizes indicating their frequency or prominence in the analyzed data.
   - Larger words mean they are mentioned more often, while smaller words are less frequent.
   - Central words like "think," "need," "come," and "read" are large, suggesting they are common in the neutral messages analyzed. These words suggest a concentration on reflection, necessity, invitation, and absorption of information.
   - The colors chosen do not evoke any strong emotional response, aligning with the theme of neutrality.
     
4. **In the [negative messages word cloud](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing)**:
   - The cloud does not provide specific contexts but gives a general overview of various words that might appear in messages with negative sentiments. 
   - It would be particularly useful in qualitative aspects of sentiment analysis, providing insights into the common vocabulary associated with negative feedback or experiences.

5. **Message length distribution by sentiment**:
   - Each [boxplot](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) shows the median length of messages (the line within each box), the interquartile range (the box itself), the range (the lines extending from the box, known as whiskers), and outliers (individual points beyond the whiskers). 
   - Messages with Neutral sentiment have a very tight interquartile range and fewer extreme outliers compared to Positive and Negative sentiments. 
   - Positive and Negative sentiment messages show more variability in length with some extreme outliers observed. 
   - The title "Message Length Distribution by Sentiment" tells us these [plots](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) are meant to compare message lengths across different sentiment classifications.
     
6. **Sentiment over time**:
   - The [line chart](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) is titled "Sentiment Over Time" and shows how the count of messages for each sentiment category (Neutral, Negative, and Positive) changes over the years from 2005 to 2015. 
   - Each sentiment is represented by a line of a different color, indicated in the legend. 
   - There's a visible overall trend with Neutral sentiment starting from a high count and decreasing dramatically over time. 
   - Positive sentiment shows some fluctuations with spikes in certain years, while Negative sentiment remains relatively low and stable with a few minor peaks. 
   - This [chart](https://docs.google.com/presentation/d/1DsxW3_5Ye-w6xH6-VxXQbl0s5O6yMnVCSZ7kwfJqzy4/edit?usp=sharing) helps analyze the trend of sentiments in messages over a decade.


In summary, while the sentiment analysis project serves as a solid foundation for understanding and employing machine learning techniques in text analysis, there are numerous avenues for future enhancement and application. This adaptability will allow the project to evolve and remain useful as new technologies and data sources emerge.
