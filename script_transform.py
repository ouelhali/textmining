from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from colorama import Fore
from IPython.display import IFrame
from tqdm import tqdm

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .master("local")\
        .appName("TfIdf Example")\
        .getOrCreate()
      
    with open('data_km.json') as f:
      data = json.load(f)
    
    sqlContext = SQLContext(spark)
    tech_text = spark.read.json('data_km.json').rdd
    number_of_docs = tech_text.count()
    print(number_of_docs)
    
    def clean(text):
      stopset = set(nltk.corpus.stopwords.words('english'))
      stemmer = nltk.PorterStemmer()
      tokens = tokenize(text) 
      
      clean = [token.lower().rstrip('.') for token in tokens if token.lower() not in stopset and len(token) > 2]
      final = [stemmer.stem(word) for word in clean]
      return final
    
    
    def lema(text):
      stopset = set(nltk.corpus.stopwords.words('english'))
      lmtzr = WordNetLemmatizer()
      tokens = tokenize(text) 
      
      clean = [token.lower().rstrip('.') for token in tokens if token.lower() not in stopset and len(token) > 2]
      final = [lmtzr.lemmatize(word) for word in clean]
      
      
      return final
    
    
    
    def tokenize(s):
      
      return re.split("\\W+", s.lower())
    #We Tokenize the text
    
    tokenized_text = tech_text.map(lambda (text,title): (title, lema(text)))
    #Count Words in each document
    term_frequency= tokenized_text.flatMapValues(lambda x: x).countByValue()

    term_frequency = tokenized_text.flatMapValues(lambda x: x).countByValue()
  
    document_frequency = tokenized_text.flatMapValues(lambda x: x).distinct()\
                        .map(lambda (title,word): (word,title)).countByKey()

    
    import numpy as np


    def tf_idf(number_of_docs, term_frequency, document_frequency):
      result = []
      for key, value in tqdm(term_frequency.items()):
          doc = key[0]
          term = key[1]
          df = document_frequency[term]
          if (df>0):
            tf_idf = float(value)*np.log(number_of_docs/df)
        
          result.append({"doc":doc, "score":tf_idf, "term":term})
      return result



    tf_idf_output = tf_idf(number_of_docs, term_frequency, document_frequency)
    

    #tf_idf_output[:10]
    tfidf_RDD = spark.sparkContext.parallelize(tf_idf_output).map(lambda x: (x['term'],(x['doc'],x['score']) )) # the corpus with tfidf scores
    
    print(type(tfidf_RDD))
    
    #print(type(tfidf_RDD))
    #df_tfidf = spark.read.parquet("tfidf_parquet")
    #tfidf_RDD= df_tfidf.rdd.map(lambda x: (x['term'],(x['doc'],x['score']) ))
    import datetime
    
    def search(query, topN):
      print(datetime.datetime.now())
      tokens = spark.sparkContext.parallelize(lema(query)).map(lambda x: (x,1) ).collectAsMap()
      bcTokens = spark.sparkContext.broadcast(tokens)

  #connect to documents with terms in the Query. to Limit the computation space  
      joined_tfidf = tfidf_RDD.map(lambda (k,v): (k,bcTokens.value.get(k,'-'),v) ).filter(lambda (a,b,c): b != '-' )
  
  #compute the score using aggregateByKey
      scount = joined_tfidf.map(lambda a: a[2]).aggregateByKey((0,0),
      (lambda acc, value: (acc[0] +value,acc[1]+1)),
      (lambda acc1,acc2: (acc1[0]+acc2[0],acc1[1]+acc2[1])) )
  
      scores = scount.map(lambda (k,v): ( v[0]*v[1]/len(tokens), k) ).top(topN)
      print(datetime.datetime.now())
      return scores
   
    def get_answer(query, document):
      stops = set(stopwords.words('english'))

      with open('data_km.json') as f:
        data = json.load(f)

      for i in tqdm(range(len(data))):
        if data[i]['title'] == document:
          text_sample=data[i]['text'].lower()
  
      text_sentences= sent_tokenize(text_sample)


      text_sentences.insert(0, query)
      
    #n = 6
    #sixgrams = ngrams(sentence.split(), n)
    #sentences2= word_tokenize(sentence)
    #print(sentences2)

    #output = [w for w in sentences2 if not w in stops]

    #print(output)
      lmtzr = WordNetLemmatizer()
      stemmer = nltk.PorterStemmer()
      origin_sentences = text_sentences[:]
      for index in tqdm(range(len(text_sentences))):
        untok = word_tokenize(text_sentences[index])
        untok = [word for word in untok if word not in stops]
        untok = [lmtzr.lemmatize(word) for word in untok]
        text_sentences[index] = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in untok]).strip()



      tfidf_vectorizer = TfidfVectorizer()
      tfidf_matrix = tfidf_vectorizer.fit_transform(text_sentences)

      value = (cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))
      value[0][0] = 0 
      return origin_sentences[value.argmax()]
    
    def get_page(answer, file_pdf):
      with open('page_index.json') as filepage:
        file_index = json.load(filepage)
      flag = False
      
      name_doc = '/home/oelhali'+file_pdf
      queries= [answer]


      for i in tqdm(range(len(file_index))):
  
        if file_index[i]['title'] == name_doc:
          flag= True
          output = file_index[i]

          for j in range(len(output)-1):
            queries.append(output[str(j)].lower())
          
            tfidf_vectorizer = TfidfVectorizer()
            para_matrix = tfidf_vectorizer.fit_transform(queries)

            pages = (cosine_similarity(para_matrix[0:1], para_matrix))
            pages[0][0] = 0 
            num_page= pages.argmax()
      if flag == False:
        return "cant get page"
      
      return 'https://apps.faurecia/sites/techknow/Lists/KM' + document[11:]+'#page='+str(num_page)

    def score_check(score):
      if score >= 350:
        return "Good"
      elif score< 350 and score >= 150:
        return "Average"
      else:
        return "Bad"
        
    
    import csv
    import collections
    from collections import Counter

    from sklearn.feature_extraction.text import CountVectorizer



    def get_top_n_words(corpus, n=None):
  
    
      vec = CountVectorizer().fit(corpus)
      bag_of_words = vec.transform(corpus)
      sum_words = bag_of_words.sum(axis=0) 
      words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
      words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
      return words_freq[:n]
  


    def get_cat(paths):
      
      
      link=[]
      
      for i in range(len(paths)):
        tmp_link= re.findall('/sharepoint/(.+)/', paths[i][1])
        link.append('sites/techknow/Lists/KM/'+tmp_link[0])
      #print(link)
      liste=[]

      with open('km_cat.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=';')
        liste = map(tuple, reader)
    
      ensemble= []
      for item in link:
        for i in range(len(liste)):
          if item == liste[i][5]:
            cat= liste[i][4].replace(';','')
            clean= cat.split('#')
            clean= [s for s in clean if not re.search(r'\d',s)]
            ensemble.extend(clean)
      print(ensemble)
      clean_cat=[]
      [clean_cat.append(x) for x in ensemble if x not in clean_cat]
      print(clean_cat)
      ensemble =  Counter(ensemble)
      
      list_cat= ensemble.most_common(3)
      #return list_cat[0][1], list_cat[0][2], list_cat[0][3]
      return [row[0] for row in list_cat]
    
    
    
    
    print('######################## RESULTS ##############################')
    
    
    
    
    #Example Questions
    # what are the mechanical properties of NAFIlean
    # what is IMP ?
    def quest(args):
      par_test= search(args["a"],5 )
      document_f= par_test[0][1]
      lien= get_page(get_answer(args["a"], document_f),document_f)
      link_t1 = "<a href='{href}'> Click here to see the answer</a>"
      html2 = HTML(link_t.format(href=lien))
      
      return html2

    
    #Put your Question here
    quest_query="how to solve a foam leakage ?"
    
    

    
    ranks= search(quest_query,5 )
    print(ranks)
    #print('       ')
    print(Fore.RED + "Title", ranks[0][1])
    document= ranks[0][1]
    
    print(Fore.GREEN + "Rating : ",score_check(int(ranks[0][0])))
    
    
    from IPython.display import IFrame

    from IPython.display import display, HTML

    link_t = "<a href='{href}'> Click here to see the answer</a>"
    result_file = get_page(get_answer(quest_query, document),document)
    html = HTML(link_t.format(href=result_file))
    display(html)
    print(ranks)
    print("Suggested categories:" , get_cat(ranks))
    
    print("                ")
    
    
    
    """
     
    with open("question_test.txt",'rb') as file_quest:
      for line in file_quest:
        print("Question : ", line)
        ranks= search(line,5 )
        print(ranks)
        #print('       ')
        print("Title", ranks[0][1])
        document= ranks[0][1]
        print(ranks[0][1])
        print("Rating : ",score_check(int(ranks[0][0])))
        print(get_page(get_answer(line, document),document))
        print(get_cat(ranks))
        #print('*********************')
      
      """

 
    #tokenizer = Tokenizer(inputCol="text", outputCol="words")
    #wordsData = tokenizer.transform(sentenceData)
 
    #hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    #featurizedData = hashingTF.transform(wordsData)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
 
    #idf = IDF(inputCol="rawFeatures", outputCol="features")
    #idfModel = idf.fit(featurizedData)
    #rescaledData = idfModel.transform(featurizedData)
 
    #rescaledData.select("text", "features").show()

    #rescaledData.select("text", "features").show()
