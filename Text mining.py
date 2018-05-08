import re
import jieba
#import jieba.analyse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data =  pd.read_excel("/transfer/LQ rawdata.xlsx")
data.head()
user_dict = "/transfer/LQ key.txt"
stop_words_path = "/transfer/stop_words"
with open(stop_words_path,'r') as f:
	stop_words = [line.strip() for line in f.readlines()]
jieba.load_userdict(user_dict)

x = data['PROD_DESC_RAW']+data['ATTRIBUTE']
#list = jieba.cut(x)
#tags = jieba.analyse.extract_tags(str(x),topK=20)

x_corpus = []
for i in x:
	i = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*【】|:≤\-（）<]+|[\d+]{5,}"," ",i)
	#print(i)
	seg = jieba.cut(i)
	#print(seg)
	for word in seg:
		if word not in stop_words:
			#print(word)
			x_corpus.append("".join(word))
x_corpus = [" ".join(x_corpus)]
#cut data into one string.

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(x_corpus)
kyws = pd.DataFrame(tfidf.toarray()).T
#get tfidf for string, and put them into a dataframe.
kyws.columns = ['weight']
#only tfidf values calcaluted, therefore, name the values as weight
word=vectorizer.get_feature_names()
kyws['word']=word
#get the
kyws = kyws.sort_values(by="weight",ascending=False)
#reorder the keywords by
kyws['segment']=''
kyws.loc[kyws['word'].str.contains("德国|法国|比利时|俄罗斯|新西兰|新疆|青岛|智利|日本|南非|西班牙|捷克|中国|英国|墨西哥|意大利|进口|荷兰"),['segment']]='Origin'
kyws.loc[kyws['word'].str.contains("红酒|葡萄酒|啤酒|威士忌|力娇酒|朗姆酒|洋酒|利口酒|鸡尾酒"),['segment']]='Category'
#kyws.to_excel("/transfer/feature dict.xlsx",sheet_name="dict")
#Generate the dictionary file for all titles. Ranked by tfidf.



data ['words']=""
for i in range(len(data)):
		line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*【】|:≤\-（）<]+|[\d+]{5,}", " ",data['PROD_DESC_RAW'][i]+data['ATTRIBUTE'][i]).strip()
		#print(line)
		seg = jieba.cut(line)
		words = []
		for word in seg:
			if (word not in stop_words) and (word !=" "):
				words.append("".join(word))
		data.loc[i,'words']=" ".join(words)
x = data['words']
y = data['PACKSIZE']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train_corpus = []
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(x_train_corpus)
x_train_matrix = pd.DataFrame(tfidf.toarray())
x_train_matrix.columns =vectorizer.get_feature_names()
clf = KNeighborsClassifier()
#clf = SVC()
clf.fit(x_train_matrix,y_train)