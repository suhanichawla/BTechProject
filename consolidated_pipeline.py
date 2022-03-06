

"""## IMPORT AND CONFIG"""

# # TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

#BERT
from sentence_transformers import SentenceTransformer
import json

# UNIVERSAL ENCODER
import tensorflow as tf
import tensorflow_hub as hub

# # DOC2VEC
import pickle
from nltk.tokenize import word_tokenize

# UTILS
import networkx as nx
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedShuffleSplit, HalvingGridSearchCV
from collections import Counter


# NODE2VEC
from node2vec import Node2Vec
import collections
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

config = {
  'corpus':"./Subdatasets/5050/subdataset_",
  'saveFolder':"./",
  'similarityMetric': ["cosine","dot"], #emneddings must be in datasets folder ['dot','cosine']
  'nodeIDs':"./Subdatasets/5050/subdataset_", #node ids of subdataset
  'subdataset':"./Subdatasets/5050/subdataset_", #{id:{mentions:[], urls:[], label:T/F, idx:int}}
  'normalization': True,
  'min-max-file':"./Subdatasets/5050/subdataset_",
  'similarityModel': ['TFIDF', 'BERT', 'UE','DOC2VEC'],
  'modelURL': ["", 'bert-mean-nli-tokens', "", "/content/drive/MyDrive/datasets/model/d2v.model"]
}

def save_embeddings(i, sim_model_idx):
  file = open(config['corpus']+str(i)+"_corpus.json")
  corpus = json.load(file)
  file.close()

  if sim_model_idx == 0:
    print("TFIDF generating embeddings")
    tfidf_vectorizer = TfidfVectorizer()
    embeddings = tfidf_vectorizer.fit_transform(corpus).toarray()
    embeddings = embeddings.tolist()
  elif sim_model_idx == 1:
    print("BERT generating embeddings")
    model = SentenceTransformer(config['modelURL'][sim_model_idx])
    embeddings = model.encode(corpus)
    embeddings = embeddings.tolist()
  elif sim_model_idx == 2:
    print("UE generating embeddings")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(corpus)
    embeddings = embeddings.numpy()
    embeddings = embeddings.tolist()
  elif sim_model_idx == 3:
    print("D2V generating embeddings")
    model = pickle.load(open(config['modelURL'][sim_model_idx], 'rb'))
    embeddings = []
    for sentence in corpus:
      v1 = model.infer_vector(word_tokenize(sentence))
      embeddings.append(v1.tolist())

  json_object = json.dumps(embeddings, indent=4, sort_keys=True, default=str)
  print(config['saveFolder']+config['similarityModel'][sim_model_idx] , config['similarityModel'][sim_model_idx] )
  with open(config['saveFolder']+config['similarityModel'][sim_model_idx] +"/5050/subdataset_"+str(i)+"_"+config['similarityModel'][sim_model_idx]+"_embeddings.json", "w") as outfile:
    outfile.write(json_object)
  print("embeddings saved")

def consolidated_embeddings(i):
  consolidated_embedding = []
  for embed in config["similarityModel"]:
      file = open(config['saveFolder'] + embed +"/5050/subdataset_" + str(i) +"_"+embed+ "_embeddings.json")
      text_embeddings = json.load(file)
      file.close() 

      if consolidated_embedding == []:
        consolidated_embedding = text_embeddings
      else:
        i = 0
        for embedding in consolidated_embedding:
          consolidated_embedding[i] = embedding + text_embeddings[i]
          i += 1
  json_object = json.dumps(consolidated_embedding, indent=4, sort_keys=True, default=str)
  with open(config['saveFolder']+"CONSOLIDATED/5050/subdataset_"+str(i) + "_consolidated_embeddings.json", "w") as outfile:
    outfile.write(json_object)


####### HELPER FUNCTIONS
def attr_intersection_weight(list1, list2):
  return len(set(list1).intersection(list2))

def getNormalised(value, max, min):
  return (value - min) / (max - min)

def findSimilarity(text1, text2, metric):
  if metric=='cosine':
      score = cosine_similarity(
          [text1],
          [text2]
      )
      return score[0][0]
  elif metric=='dot':
    return np.dot(text1,text2)


####### MAIN FUNC
def get_truth_false_scores(i, metric):
  file = open(config['nodeIDs']+str(i)+"_nodeIDs.json")
  nodeIDs = json.load(file)
  file.close()

  file = open(config['subdataset']+str(i)+"_nodes.json")
  nodesdata = json.load(file)
  file.close() 
  
  file = open(config['saveFolder']+"CONSOLIDATED/5050/subdataset_"+str(i)+"_consolidated_embeddings.json", 'rb')
  embeddings = json.load(file)
  file.close()

  file = open(config['min-max-file']+str(i)+"_min_max_info.json")
  minmax = json.load(file)
  file.close()

  truth = []
  false = []
  for node in nodeIDs:
    if nodesdata[node]["label"]:
      truth.append(node)
    else:
      false.append(node)

  
    
  truth_pairs = combinations(truth, 2)
  false_pairs = combinations(false, 2)
  pairs = combinations(nodeIDs, 2)

  truth_scores = []
  false_scores = []
  scores = []

  co_hashtag_truth = []
  co_hashtag_false = []
  co_hashtag_both = []

  co_mention_truth = []
  co_mention_false = []
  co_mention_both = []

  co_url_truth = []
  co_url_false = []
  co_url_both = []

  ts_alone=[]
  fs_alone=[]
  s_alone=[]

  print("truth")
  for pair in truth_pairs:
    ts = findSimilarity(embeddings[nodesdata[pair[0]]['idx']], embeddings[nodesdata[pair[1]]['idx']], metric)
    ts_alone.append(ts)
    if not config['normalization']:
      co_hashtag = attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags'])
      co_mention = attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions'])
      co_url = attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls'])
    else:
      co_hashtag = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags']),minmax['max_cohashtags'],minmax['min_cohashtags'])
      co_mention = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions']),minmax['max_comentions'],minmax['min_comentions'])
      co_url = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls']),minmax['max_courls'],minmax['min_courls'])
    co_hashtag_truth.append(co_hashtag)
    co_mention_truth.append(co_mention)
    co_url_truth.append(co_url)
    ts += (co_hashtag*0.1) + (co_mention*0.1) + (co_url*0.1)
    truth_scores.append(ts)

    
  print("false")
  for pair in false_pairs:
    fs = findSimilarity(embeddings[nodesdata[pair[0]]['idx']], embeddings[nodesdata[pair[1]]['idx']], metric)
    fs_alone.append(fs)
    if not config['normalization']:
      co_hashtag = attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags'])
      co_mention = attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions'])
      co_url = attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls'])
    else:
      co_hashtag = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags']),minmax['max_cohashtags'],minmax['min_cohashtags'])
      co_mention = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions']),minmax['max_comentions'],minmax['min_comentions'])
      co_url = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls']),minmax['max_courls'],minmax['min_courls'])
    co_hashtag_false.append(co_hashtag)
    co_mention_false.append(co_mention)
    co_url_false.append(co_url)
    fs += (co_hashtag*0.1) + (co_mention*0.1) + (co_url*0.1)
    
    false_scores.append(fs)

  print("interclass")
  for pair in pairs:
    s = findSimilarity(embeddings[nodesdata[pair[0]]['idx']], embeddings[nodesdata[pair[1]]['idx']], metric)
    #print("score for this pair", s)
    s_alone.append(s)
    if not config['normalization']:
      co_hashtag = attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags'])
      co_mention = attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions'])
      co_url = attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls'])
    else:
      co_hashtag = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags']),minmax['max_cohashtags'],minmax['min_cohashtags'])
      co_mention = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions']),minmax['max_comentions'],minmax['min_comentions'])
      co_url = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls']),minmax['max_courls'],minmax['min_courls'])
    co_hashtag_both.append(co_hashtag)
    co_mention_both.append(co_mention)
    co_url_both.append(co_url)
    s += (co_hashtag*0.1) + (co_mention*0.1) + (co_url*0.1)
    scores.append(s)
  print("returning truth false and overall scores")
  return truth_scores, false_scores, scores, co_hashtag_truth, co_hashtag_false, co_hashtag_both, co_mention_truth, co_mention_false, co_mention_both, co_url_truth, co_url_false, co_url_both, ts_alone, fs_alone, s_alone


def drawBoxPlots(truth_scores, false_scores, scores, title):
  fig1 = plt.figure(figsize =(10, 7))
  plt.boxplot([
    truth_scores, 
    false_scores, 
    scores
  ])
  plt.title(title)
  plt.show()
  plt.savefig(title+".jpg")
  print("median of truth : ", np.median(truth_scores))
  print("median of false : ", np.median(false_scores))
  print("median of interclass : ", np.median(scores))
  return np.median(scores)

def findThresh(truth_scores, false_scores):
  data1 = truth_scores
  data2 = false_scores

  count1, bins_count1 = np.histogram(data1, bins=10)
  count2, bins_count2 = np.histogram(data2, bins=10)

  pdf1 = count1 / sum(count1)
  pdf2 = count2 / sum(count2)
  
  y1 = np.cumsum(pdf1)
  y2 = np.cumsum(pdf2)

  x1 = bins_count1[1:]
  x2 = bins_count2[1:]

  mini = max(min(x1), min(x2))
  maxi = min(max(x1), max(x2))

  thresh = mini
  maxdiff = 0
  i = mini
  while(i<maxi):
    diff = np.interp(i,x1,y1) - np.interp(i,x2,y2)
    if(diff > maxdiff):
      print(diff)
      maxdiff = diff
      thresh = i
    i += 0.1

  return thresh


def split_and_predict(clf, param_grid, X, y, n, details, overall_best, output):

  rep = {
    "0": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0
    },
    "1": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0
    },
    "accuracy": 0.0,
    "weighted avg": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0
    }
  }

  # print("Splitting")
  # print(len(X))
  # print(len(y))
  
  sss = StratifiedShuffleSplit(n_splits=n, test_size=0.3, random_state=0)
  for train_index, test_index in sss.split(X, y):

    # print(len(train_index), " + ", len(test_index), " = ", len(train_index)+len(test_index))
    # print(max(train_index))
    # print(max(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    search = HalvingGridSearchCV(clf, param_grid,
                                 min_resources='exhaust', factor=3, random_state=0).fit(X_train, y_train)
    cfg = search.best_estimator_
    print("best config", cfg)
    cfg.fit(X_train, y_train)
    y_pred = cfg.predict(X_test)

    subrep = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    del subrep["macro avg"]

    rep["0"] = dict(Counter(rep["0"]) + Counter(subrep["0"]))
    rep["1"] = dict(Counter(rep["1"]) + Counter(subrep["1"]))
    rep["accuracy"] += subrep["accuracy"]
    rep["weighted avg"] = dict(Counter(rep["weighted avg"]) + Counter(subrep["weighted avg"]))
    
  for key in rep:
    if type(rep[key]) is dict:
      for param in rep[key]:
        rep[key][param] /= n
    else:
      rep[key] /= n

  details["Result"] = rep
  overall_best.append(rep["weighted avg"]["f1-score"])
  output.append(details)
  
  return cfg



def Classify(model, i, text_embeddings=False):

  file = open(config['subdataset']+str(i)+"_nodes.json")
  data = json.load(file)
  file.close() 

  labels = []

  for user in data:
    if data[user]["label"]:
      labels.append(1)
    else:
      labels.append(0)

  y = np.array(labels)
  if text_embeddings:
    X = model
  else:
    X = (model.wv.vectors)
  X = np.array(X)

  n=5

  output = []
  overall_best = []

  

####### SVM
  print("SVM Classifier")
  param_grid = {
      "C": [1, 10, 100],
      "kernel": ("linear", "rbf", "sigmoid"), 
  }
  clf = svm.SVC(class_weight="balanced", gamma="scale")
  svm_details = {}
  svm_details["Config"] = split_and_predict(clf, param_grid, X, y, n, svm_details, overall_best, output)
  
####### RF
  print("RF Classifier")
  param_grid = {
      'max_depth': [3, 5, 10],
      'min_samples_split': [2, 5, 10],
      'n_estimators': [50,100,150]
  }
  clf = RandomForestClassifier(random_state=0)
  rf_details = {}
  rf_details["Config"] = split_and_predict(clf, param_grid, X, y, n, rf_details, overall_best, output)


####### KNN
  print("KNN Classifier")
  param_grid = {
      'n_neighbors': [3, 5, 7],
      'p': [1,2,3]
  }
  clf = KNeighborsClassifier(n_jobs=-1)
  knn_details = {}
  knn_details["Config"] = split_and_predict(clf, param_grid, X, y, n, knn_details, overall_best, output)


####### ADA
  print("ADA Classifier")
  param_grid = {
      "n_estimators": [25, 50, 75],
      "learning_rate": [0.5, 1, 1.5], 
  }
  clf = AdaBoostClassifier()
  ada_details = {}
  ada_config = split_and_predict(clf, param_grid, X, y, n, ada_details, overall_best, output)
  ada_details["Config"] = ada_config


  output.append(max(overall_best))
  return output

"""### CREATING GRAPH"""

####### HELPER FUNC
def attr_intersection_weight(list1, list2):
  return len(set(list1).intersection(list2))

def getNormalised(value, max, min):
  return (value - min) / (max - min)



######### MAIN FUNC
def create_graph(threshold, cosines, i):
  file = open(config['nodeIDs']+str(i)+"_nodeIDs.json")
  nodes = json.load(file)
  file.close()

  file = open(config['subdataset']+str(i)+"_nodes.json")
  nodesdata = json.load(file)
  file.close() 
  
  file = open(config['min-max-file']+str(i)+"_min_max_info.json")
  minmax = json.load(file)
  file.close()

  # creating nodes
  print("creating nodes")
  graph = nx.Graph()
  labels={}
  for x in nodes:
      labels[x] = nodesdata[x]['label']
      graph.add_node(x)

  # Adding label for gephi
  nx.set_node_attributes(graph, labels, name="label")

  # creating edges
  print("creating edges")
  pairs = combinations(nodes, 2)
  
  pno=0
  for pair in pairs:
    if pno % 10000 == 0:
      print(pno)

    weight = cosines[pno]

    if weight > threshold:
      if not config['normalization']:
        co_hashtag = attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags'])
        co_mention = attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions'])
        co_url = attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls'])
        weight += (co_hashtag + co_mention + co_url)*0.1
      else:
        co_hashtag = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['hashtags'], nodesdata[pair[1]]['hashtags']),minmax['max_cohashtags'],minmax['min_cohashtags'])
        co_mention = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['mentions'], nodesdata[pair[1]]['mentions']),minmax['max_comentions'],minmax['min_comentions'])
        co_url = getNormalised(attr_intersection_weight(nodesdata[pair[0]]['urls'], nodesdata[pair[1]]['urls']),minmax['max_courls'],minmax['min_courls'])
        weight += co_hashtag + co_mention + co_url

      graph.add_weighted_edges_from([(pair[0], pair[1], weight)])

    pno+= 1

  #nx.write_edgelist(graph, config['saveFolder']+config['similarityModel']+"/5050/subdataset_"+str(i)+ "_weighted_edgelist_"+str(threshold)+".txt", delimiter=' ')
  num = round(threshold,5)
  nx.write_gexf(graph, config['saveFolder']+"CONSOLIDATED/5050/subdataset_"+str(i)+ "_weighted_graph_"+str(num)+".gexf")

  # edges = json.dumps(list(graph.edges(data=True)), indent=4, sort_keys=True, default=str)
  # with open(config['saveFolder']+config['similarityModel'][simidx]+"/5050/subdataset_"+str(i)+ "_edges_"+str(num)+".json", "w") as outfile:
  #   outfile.write(edges)

  print("graph created for threshold "+str(threshold), graph)
  return graph

def node2vec(sim_model, graph, best_model_config, walk_len, dimensions, pq, threshold, i):
  for dim in dimensions:
    for len in walk_len:
      for pq_test in pq:
        p_test = pq_test[0]
        q_test = pq_test[1]

        print("Running for dimension "+ str(dim) + ", p "+str(p_test)+", q "+str(q_test)+", walk length "+str(len))
        try:
          model = pickle.load(open(config['saveFolder'] + sim_model + "/5050/subdataset_"+str(i)+"_n2v_model_" + 
                                   str(dim)+"_" + str(len)+"_" + str(p_test) + "_" + str(q_test)+"_"+str(threshold), 'rb'))
          print("Found existing model")
        except Exception as e:
          print("error", e)
          model = None

        if not model:
          print("Didn't find model, creating one")
          node2vec = Node2Vec(graph, walk_length=len, dimensions=dim, p=p_test, q=q_test, workers=4)  # Use temp_folder for big graphs
          model = node2vec.fit(window=10, min_count=1, batch_words=4)
          
          model.wv.save_word2vec_format(config['saveFolder'] + sim_model + "/5050/subdataset_"+str(i)+ "_n2v_embeddings_" + str(dim)+"_" + str(len)+"_" + str(p_test) + "_" + str(q_test)+"_"+str(threshold))
          
          model.save(config['saveFolder'] + sim_model + "/5050/subdataset_"+str(i)+ "_n2v_model_" + str(dim)+"_" + str(len)+"_" + str(p_test) + "_" + str(q_test)+"_"+str(threshold))
          print("saved model")
        c_output = Classify(model, 0)
        if best_model_config["f1-score"] < c_output[4] :
          best_model_config["Dimension"] = dim
          best_model_config["Walk length"] = walk_len
          best_model_config["p"] = p_test
          best_model_config["q"] = q_test
          best_model_config["Classifiers"] = c_output
          best_model_config["Sim Model"] = sim_model
          best_model_config["f1-score"] = c_output[4]

          best_model = json.dumps(best_model_config, indent=4, sort_keys=True, default=str)
          with open(config['saveFolder'] + sim_model + "/5050/subdataset_" + str(i) + "_best_model_config_"+str(threshold)+".json", "w") as outfile:
            outfile.write(best_model)


def run_node2vec_for_graphs(thresholds, best_model_config, i):
  for thresh in thresholds:
    print("Running for iteration "+str(thresh))
    num = round(thresh, 5)
    graph = nx.read_gexf(config['saveFolder']+"CONSOLIDATED/5050/subdataset_"+str(i)+ "_weighted_graph_"+str(num)+".gexf")
    
    walk_len = [60,80,100] 
    pq = [[1,1], [2,1], [1,2]]
    dim = [64, 128, 256]
    # walk_len = [1]
    # pq = [[1,1]]
    # dim = [64]

    model = node2vec("CONSOLIDATED", graph, best_model_config, walk_len, dim, pq, thresh, i)

    print("Finished for iteration "+str(thresh))


"""# INDEX BOX"""
##
import warnings
warnings.filterwarnings('always')

simmodels = ['TFIDF','BERT', 'UE', 'DOC2VEC']
for y in range(18): #!change to 18
# for y in range(17):
  
    consolidated_embeddings(y)
    truth_scores_dot, false_scores_dot, scores_dot, co_hashtag_truth_dot, co_hashtag_false_dot, co_hashtag_both_dot, co_mention_truth_dot, co_mention_false_dot, co_mention_both_dot, co_url_truth_dot, co_url_false_dot, co_url_both_dot, ts_alone_dot, fs_alone_dot, s_alone_dot = get_truth_false_scores(y, 'dot')
    truth_scores_cosine, false_scores_cosine, scores_cosine, co_hashtag_truth_cosine, co_hashtag_false_cosine, co_hashtag_both_cosine, co_mention_truth_cosine, co_mention_false_cosine, co_mention_both_cosine, co_url_truth_cosine, co_url_false_cosine, co_url_both_cosine, ts_alone_cosine, fs_alone_cosine, s_alone_cosine = get_truth_false_scores(y, 'cosine')

    thresh1 = 0
    s_alone = []
    ts_alone = []
    fs_alone = []
    #print(np.median(s_alone_dot), np.median(s_alone_cosine))
    if(np.median(s_alone_dot) > np.median(s_alone_cosine)):
        thresh1 = np.median(s_alone_dot)
        s_alone = s_alone_dot
        ts_alone = ts_alone_dot
        fs_alone = fs_alone_dot
    else:
        thresh1 = np.median(s_alone_cosine)
        s_alone = s_alone_cosine
        ts_alone = ts_alone_cosine
        fs_alone = fs_alone_cosine

    thresh2 = findThresh(ts_alone, fs_alone)

    threshArr = [thresh1, thresh2]
    #print(threshArr)
    for z in threshArr:
        print("salone len", len(s_alone))
        graph = create_graph(z, s_alone, y)


    best_model_config = {
        "Dimension": 0,
        "Walk length": 0,
        "p": 0,
        "q": 0,
        "Classifiers": "",
        "Sim Model": "consolidated",
        "f1-score": 0,
    }
    run_node2vec_for_graphs(threshArr, best_model_config, 0)