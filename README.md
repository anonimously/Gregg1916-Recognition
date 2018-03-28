# Gregg1916-A-Corpus-and-a-Novel-Approach-for-Optical-Gregg-Shorthand-Recognition 

This repository collects our code and corpus for Gregg shorthand recognition.  

Gregg shorthand is one of the shorthand languages that allows fast recording of information. Check this page for the details:  
https://en.wikipedia.org/wiki/Gregg_shorthand

The corpus in .\data is produced from a shorthand dictionary. Each word in the dictionary gets one image.  

The method admits some novelty by assembling methodologies from NLP and CV. The ingredients include:  
* A CNN-based feature extractor;  
* A character level RNN language model; 
* Image augmentation including scaling, shifting and rotation; 
* Information retrieval with Levenshtein distance;  
* The use of bleu score for information retrieval;  
* A 'bidirectionally weighted bleu' which takes into account two candidates.

The architecture goes as follows:  
* Feature extractor: a CNN extracts features from images;
* Decoder: initialized by the extracted features, two RNNs decodes from the forward and the backward direction respectively to yield a forward hypothesis and a backward hypothesis;
* Word retrieval: combining both hypotheses to retrieve the most promising word from the full vocabulary.  

A more fine-grained technical report will be added soon.
