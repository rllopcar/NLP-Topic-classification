# Profile-based retrieval

The student is required to create a method based in the space vector model to deliver small text snippets to different users depending on their profile.

For instance, let us suppose that we have 4 different users: the first one being interested in politics and soccer, the second in music and films, the third in cars and politics and the fourth in soccer alone.

An incoming document targeted at politics should be delivered to users 1 and 3, while a document on soccer should be delivered to users 1 and 4. 

**Students must submit a written report no longer than 15 pages** explaining <u>the method used to encode both the documents and the user profiles</u>, together with the algorithm used to process the queries (the more efficient, the better). 

**The written report, which is mandatory will provide a grade of 8 (out of 10 points) maximum.**

**To obtain the maximum grade (10 points out of 10), the student must provide a solid implementation of the proposed method in any programming language.**

 The instructor recommends students to choose the Python programming language or Java since there are plenty of useful code snippets out there to help implement the required functionalities. If the student decides to submit the optional part, all the required stuff to execute the program must be provided.
 

## NLP basic concepts

- **Lemmatization** The task of removing inflectional endings only and to return the base dictionary form of a word which is also known as a lemma.
- **Part of speech** is a category of words (or, more generally, of lexical items) which have similar grammatical properties.
- **Part of speech tagging** Given a sentence, determine the part of speech for each word.
- **Steeming** The process of reducing inflected (or sometimes derived) words to their root form. (e.g. "close" will be the root for "closed", "closing", "close", "closer" etc).
- **Bag of words** or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms. It involves two things:
	- A vocabulary of known words
	- A measure of the presence of known words
	-  The model is only concerned with whether known words occur in the document, not where in the document.
- **N-gram:**An N-gram is an N-token sequence of words: a 2-gram (more commonly called a bigram) is a two-word sequence of words like “please turn”, “turn your”, or “your homework”, and a 3-gram (more commonly called a trigram) is a three-word sequence of words like “please turn your”, or “turn your homework".
- **TF-ID(Term Frequency – Inverse Document Frequency)**
	- Term Frequency: is a scoring of the frequency of the word in the current document.
	- Inverse Document Frequency: is a scoring of how rare the word is across documents.


There are simple text cleaning techniques that can be used as a first step, such as:

- Ignoring case
- Ignoring punctuation
- Ignoring frequent words that don’t contain much information, called stop w ords, like “a,” “of,” etc.
- Fixing misspelled words.
- Reducing words to their stem (e.g. “play” from “playing”) using stemming algorithms.

## Machine learning for Text classification

- **Naive Bayes:** is a family of statistical algorithms we can make use of when doing text classification. Naive Bayes is based on Bayes’s Theorem, which helps us compute the conditional probabilities of occurrence of two events based on the probabilities of occurrence of each individual event.
- **Support Vector Machine(SVM):**In short, SVM takes care of drawing a “line” or hyperplane that divides a space into two subspaces: one subspace that contains vectors that belong to a group and another subspace that contains vectors that do not belong to that group. Those vectors are representations of your training texts and a group is a tag you have tagged your texts with.
### - **Deep learning:** The two main deep learning architectures used in text classification are Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Text processing
The process to follow in order to transform raw text from human language to machine-readable format for further processing. Those steps are usually the following ones:
- All letter to lower or upper case.
- Converting numbers into words or removing numbers.
- Removing pucntuation, accent marks and other diacritics.
- Removing white spaces
- Tokenization
- Expanding abbreviations.
- Removing stop words, sparse terms, and particular words.
- Text canonicalization.




	
	
	
CBOW
Skip gram
TOPIC classification of news

**Sources**

- https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- http://qwone.com/~jason/20Newsgroups/
- **text classification with Keras-->**https://github.com/scikit-learn/scikit-learn/tree/master/doc/tutorial/text_analytics
- https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/
- https://nlp.stanford.edu/fsnlp/
- https://www.youtube.com/results?search_query=nlp+processing+python+nltk
- http://www.nltk.org/book_1ed/ch01.html
- http://www.nltk.org/book/ch01.html
- file:///Users/robertollopcardenal/Desktop/A%CC%81REAS%20DE%20APRENDIZAJE/machine_learning/Natural%20Language%20Processing%20Recipes%20-%20Unlocking%20Text%20Data%20with%20Machine%20Learning%20and%20Deep%20Learning%20using%20Python.%20%20Apress%20(2019).pdf
- https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
- http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
- https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
- https://en.wikipedia.org/wiki/Bag-of-words_model
- **Text classification: Comprehensive guide-->**https://monkeylearn.com/text-classification/
- **PAPER-->**http://cs229.stanford.edu/proj2018/report/183.pdf 
- https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
- **BBC API-->**ttps://newsapi.org/
- **Wor2Vec tutorials-->** https://www.tensorflow.org/tutorials/representation/word2vec
- **Wor2Vec tutorials-->** http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
- **Word2Vec Tutorial-->** https://arxiv.org/abs/1301.3781#
- **Gentle introduction to Doc2Vec-->** https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
- **Word2Vec and FastText Word Embedding with Gensim-->** https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c
- **Word2vec colab theory-->**https://chainer-colab-notebook.readthedocs.io/en/latest/notebook/official_example/word2vec.html
- **Word embeddings crash course-->** https://www.datascience.com/resources/notebooks/word-embeddings-in-python
- **Word embeddings exploratio -->** https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795
- **Multiclass text classification -->** https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
- **Surprising findings in text classification-->**https://towardsdatascience.com/surprising-findings-in-document-classification-7a79e30f1666
- **Word2vec pipeline text classification-->**https://fzr72725.github.io/2018/01/14/genism-guide.html
- **Gensim documentation-->**https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument
- **Twitter text classification Part1-->** https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90
- **Multi-Class Text Classification with Doc2Vec & Logistic Regression-->**https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
- **Text classificatio: Deep learning best practices-->** https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
- **Online news classification using Deep Learning Technique-->**https://pdfs.semanticscholar.org/44e7/01c61381ea208c468ccc7fd6cff1c7bba447.pdf
- **Real-time news classifier-->** https://cs.nyu.edu/courses/spring17/CSCI-GA.3033-006/final_projects/realtime_news_classifier.pdf	
- **Word2vec applied to Recommendation: Hyperparameters Matter-->**https://arxiv.org/pdf/1804.04212.pdf