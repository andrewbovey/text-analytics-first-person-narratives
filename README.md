# DS 5559: Exploratory Text Analytics
This repo contains the Data Products for the DS5559 Final Project. The entire corpus is available here, and the processing of those individual documents is repeatable by running 'OHCO_processing.py'  
  
Keeping the dataframe from OHCO_processing.py in your global environment allows you to move on and run HCA_PCA.py and Sentiment.py as well. These generate extensions from the processed text dataset, as well as a clustering tree visualization of chapter similarity based upon the Principle Components.  

## Data
Each source text document is located in the 'texts' folder.  
OHCO Tokenization of entire corpus as CSV: https://virginia.box.com/s/xctvq3a4zig0rcg28ka8qp3r67fgm70d  
Chunked by Chapters as CSV: https://virginia.box.com/s/djcnv4l49bwozaawjpepqsdzni578c5v  
Sentiment by Chapters as CSV: https://virginia.box.com/s/2ledzyfecl86ffkoo583j1ksum26v5i4  
Vocab for entire corpus: vocab.csv

## Processing to OHCO Format
OHCO_processing.py

## Principle Components
HCA_PCA.py

## Sentiment (VADER)
Sentiment.py

## Visualizations
All chapters (hard to read): https://virginia.box.com/s/d7k225tey1ahvhwvob7f1fazq7tgh236  
Sample of chapters (readable): https://virginia.box.com/s/qujjwli99jyk4tb6133gaho8ouhl1hkt
