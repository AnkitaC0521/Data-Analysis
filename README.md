
Climate Change Text Analysis Project  

This project analyses climate-related text data (NASA dataset) using Natural Language Processing (NLP) techniques.  
It performs sentiment analysis to capture public perception and topic modelling to uncover trending themes in climate discussions. 

Features
- Data pre-processing: text cleaning, tokenization, stop word removal, lemmatization  
- Sentiment analysis using **NLTK’s VADER**  
- Topic modelling with Genism’s LDA  
- Visualizations: sentiment distribution (bar chart) & topic word clouds  
- Insights into public perception of climate change  

 Tech Stack
- Language: Python 3.11  
- Libraries: Pandas, NLTK, Genism, Matplotlib, and WordCloud  
- Techniques: NLP, Sentiment Analysis, Topic Modelling (LDA)  

Workflow
1. Data Preprocessing
   - Remove URLs, mentions, hashtags, punctuation, and numbers  
   - Convert to lowercase  
   - Tokenize & remove stop words  
   - Lemmatize words  

2. Sentiment Analysis 
   - Classify text into Positive, Negative, Neutral  
   - Visualize sentiment distribution  

3. Topic Modelling  
   - Use **LDA** to discover key themes  
   - Generate word clouds for each topic  

4. Results & Visualization 
   - Bar charts for sentiment analysis  
   - Word clouds for topics  


Example Output
- Sentiment Distribution:  
  - Positive: 45%  
  - Neutral: 35%  
  - Negative: 20%  

- Example Topics: 
  - Topic 1: climate, warming, temperature, change  
  - Topic 2: renewable, solar, energy, future  
  - Topic 3: sea, level, ice, melting  

 Project Structure
ClimateProject/
│-- Climate.py # Main script
│-- climate_nasa.csv # Dataset
│-- README.md # Documentation

Applications
Understand public perception of climate change
Identify trending topics in environmental discussions
Support policy-making, research, and awareness campaigns

Future Improvements

Add deep learning-based sentiment analysis (BERT, RoBERTa)
Build a web dashboard using Streamlit / Flask
Extend dataset with real-time tweets/news scraping

