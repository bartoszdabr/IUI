# -*- coding: iso-8859-2 -*-

from re import X
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

global cluster_number;
cluster_number = 19


def create_list_of_additial_stop_words(num_of_words_from_category):
    df = pd.read_csv('data/result.csv', usecols=['prawo cywilne'])
    pc = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo administracyjne'])
    pa = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo farmaceutyczne'])
    pf = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo pracy'])
    ppr = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo medyczne'])
    pm = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo karne'])
    pk = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/result.csv', usecols=['prawo podatkowe'])
    ppo = get_words_from_file(df, num_of_words_from_category)
    ultimate_list = pc + pa + pf + ppr + pm + pk + ppo
    ultimate_list2 = list(dict.fromkeys(ultimate_list))
    for word in ultimate_list2:
        if word in ultimate_list:
            ultimate_list.remove(word)
    # print(list(dict.fromkeys(ultimate_list)))
    return list(dict.fromkeys(ultimate_list))


def add_all_but_most_common_words_to_documents(boosted, num_of_words_from_category):
    ultimate_list = create_list_of_additial_stop_words(num_of_words_from_category)
    f = open("boosteddata2.csv", 'w', encoding='iso-8859-2')
    f.write('boosted,mock \n')
    temp = []
    for sentence in boosted:
        sentence.replace('\n', ' ')
        sentence += sentence + sentence + sentence
        if isinstance(sentence, str):
            for word in ultimate_list:
                if word in sentence:
                    temp.append(word)
                    for w in temp:
                        sentence = sentence.replace(w, '')
                        sentence += " " + w
            temp = []
            sentence += "\n"
            f.write(sentence)
    f.close()


def create_list_of_unique_words(num_of_words_from_category):
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo cywilne'])
    pc = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo administracyjne'])
    pa = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo farmaceutyczne'])
    pf = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo pracy'])
    ppr = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo medyczne'])
    pm = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo karne'])
    pk = get_words_from_file(df, num_of_words_from_category)
    df = pd.read_csv('data/unique_result_tuples.csv', usecols=['prawo podatkowe'])
    ppo = get_words_from_file(df, num_of_words_from_category)
    ultimate_list = pc + pa + pf + ppr + pm + pk + ppo
    return ultimate_list


# new stop words
def get_words_from_file(df, num_of_words_from_category):
    string = ''
    products_list = df.values.tolist()
    for elem in products_list:
        for x in elem:
            string += x

    string = string.replace('][', ', ')
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace("'", '')
    string = string.replace("'", '')
    string = string.split(', ')
    return string[:2 * num_of_words_from_category:2]


def add_unique_words_to_documents(documents, num_of_words_to_add, num_of_words_from_category):
    ultimate_list = create_list_of_unique_words(num_of_words_from_category)
    f = open("boosteddata.csv", 'w')
    f.write('boosted,mock \n')
    for sentence in documents:
        if isinstance(sentence, str):
            for word in ultimate_list:
                if word in sentence:
                    for i in range(num_of_words_to_add):
                        sentence += " " + str(word)
            f.write(sentence)
            f.write("\n")
    f.close()


# fscore for kmeans
def calculate_fscore_kmeans(kmeans, df_processed, cluster_number):
    df_processed['cluster'] = kmeans.labels_

    clusters = df_processed.groupby('cluster')

    for cluster in clusters.groups:
        f = open('cluster' + str(cluster) + '.csv', 'w')  # create csv file
        data = clusters.get_group(cluster)[['sample', 'label']]  # get title and overview columns
        f.write(data.to_csv(index_label='id'))  # set index to id
        f.close()

    # print("Cluster centroids: \n")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(cluster_number):
        f = open('cluster' + str(i) + '.csv', 'r')
        Lines = f.readlines()
        f2 = open('clust' + str(i) + '.csv', 'w')
        for line in Lines:
            if line.strip():
                f2.write(line)
        f.close()
        f2.close()
        os.remove("cluster" + str(i) + ".csv")

    x = ""
    fa = open('actual.csv', 'w')
    fa.write("id,text,label,predicted\n")
    for i in range(cluster_number):
        df = pd.read_csv('clust' + str(i) + '.csv', usecols=['label'], encoding='iso-8859-2')
        f = open('clust' + str(i) + '.csv', 'r')
        type1 = df['label'].value_counts().idxmax()
        Lines = f.readlines()
        for line in Lines[1:]:
            line = line[:-1] + "," + type1 + "\n"
            fa.write(line)
        f.close()
    fa.close()

    for i in range(cluster_number):
        df = pd.read_csv('clust' + str(i) + '.csv', usecols=['label'], encoding='iso-8859-2')
        most_common_type = df['label'].value_counts().nlargest(cluster_number + 2)
        most_common_type1 = df['label'].value_counts().nlargest(1)
        # print("Cluster %d:" % i)
        # for j in order_centroids[i, :10]:  # print out 10 feature terms of each cluster
        #    print(' %s' % terms[j])
        # print(most_common_type / len(df))
        x = x + str(most_common_type1 / len(df)) + "\n"
        # print('------------')

    # print(x)
    predicted = pd.read_csv('actual.csv', usecols=['predicted'], encoding='iso-8859-2')
    actual = pd.read_csv('actual.csv', usecols=['label'], encoding='iso-8859-2')
    print("Fscore for kmeans: " + str(f1_score(actual, predicted, average='weighted')))


# fscore for dbscan
def calculate_fscore_dbscan(db, df_processed):
    labels = db.labels_
    df_processed['cluster'] = db.labels_
    clusters = df_processed.groupby('cluster')
    # print(df_processed['cluster'])

    for cluster in clusters.groups:
        f = open('dbcluster' + str(cluster) + '.csv', 'w')  # create csv file
        data = clusters.get_group(cluster)[['sample', 'label']]  # get title and overview columns
        f.write(data.to_csv(index_label='id'))  # set index to id
        f.close()

    for i in clusters.groups:
        f = open('dbcluster' + str(i) + '.csv', 'r')
        Lines = f.readlines()
        f2 = open('dbclust' + str(i) + '.csv', 'w')
        for line in Lines:
            if line.strip():
                f2.write(line)
        f.close()
        f2.close()
        os.remove("dbcluster" + str(i) + ".csv")

    fa = open('actualdb.csv', 'w')
    fa.write("id,text,label,predicted\n")
    for i in clusters.groups:
        df = pd.read_csv('dbclust' + str(i) + '.csv', usecols=['label'], encoding='iso-8859-2')
        f = open('dbclust' + str(i) + '.csv', 'r')
        type1 = df['label'].value_counts().idxmax()
        Lines = f.readlines()
        for line in Lines[1:]:
            line = line[:-1] + "," + type1 + "\n"
            fa.write(line)
        f.close()
    fa.close()

    x = ""
    for i in clusters.groups:
        df = pd.read_csv('dbclust' + str(i) + '.csv', usecols=['label'], encoding='iso-8859-2')
        most_common_type = df['label'].value_counts().nlargest(len(clusters.groups) + 2)
        most_common_type1 = df['label'].value_counts().nlargest(1)
        # print("Cluster %d:" % i)
        # print(most_common_type / len(df))
        x = x + str(most_common_type1 / len(df)) + "\n"
        # print('------------')

    # print(x)
    predicted = pd.read_csv('actualdb.csv', usecols=['predicted'], encoding='iso-8859-2')
    actual = pd.read_csv('actualdb.csv', usecols=['label'], encoding='iso-8859-2')
    print("Fscore for dbscan: " + str(f1_score(actual, predicted, average='weighted')))

    # no_clusters = len(np.unique(labels))
    # no_noise = np.sum(np.array(labels) == -1, axis=0)
    # print('Estimated no. of clusters: %d' % no_clusters)
    # print('Estimated no. of noise points: %d' % no_noise)


# draw PCA
def draw_PCA(cluster_method, pca_vecs, isKmeans, num):
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    dominant_type = []
    file = 'dbclust'
    clust_range = range(-1, num - 1)
    if (isKmeans):
        file = 'clust'
        clust_range = range(0, num)
    for i in clust_range:
        df = pd.read_csv(file + str(i) + '.csv', usecols=['label'], encoding='iso-8859-2')
        dominant_type.append(df['label'].value_counts().nlargest(1).keys()[0])
    scatter = plt.scatter(x0, x1, c=cluster_method.labels_, s=10, cmap='inferno')
    plt.legend(handles=scatter.legend_elements(num=None)[0], labels=dominant_type, loc='upper right',
               title='Dominujacy typ dokumentu w klastrze')
    plt.show()


# kmeans
def kmeans_alg(df_processed, features):
    kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=500, n_init=20, tol=0.0001,
                    random_state=42, algorithm="elkan").fit(features)

    calculate_fscore_kmeans(kmeans, df_processed, cluster_number)

    pca = PCA(n_components=2721)
    pca_vecs = pca.fit(features.toarray())
    pca_vecs = pca_vecs.transform(features.toarray())

    draw_PCA(kmeans, pca_vecs)


# dbscan get optimal epsilon
def dbscan_eps_designation(df_processed, features):
    f = open('dbdata.csv', 'w')  # create csv file
    data = df_processed  # get title and overview columns
    f.write(data.to_csv(index_label='id'))  # set index to id
    f.close()

    from sklearn.neighbors import NearestNeighbors
    from matplotlib.ticker import PercentFormatter

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)

    fig, ax = plt.subplots()

    distances = np.sort(distances, axis=0)
    y = []
    for i in range(len(distances)):
        y.append(i)
    distances = distances[:, 1]
    percent = distances.cumsum() / distances.sum() * 100

    ax.bar(df.index, distances)
    ax.set_title("Pareto Chart")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Epsilon");

    ax2 = ax.twinx()
    ax2.plot(y, percent, color="red", marker="D", ms=7)
    ax2.axhline(80, color="orange", linestyle="dashed")
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylabel("Cumulative Percentage");
    plt.show()


# dbscan
def dbscan_alg(df_processed, features):
    f = open('dbdata.csv', 'w')  # create csv file
    data = df_processed  # get title and overview columns
    f.write(data.to_csv(index_label='id'))  # set index to id
    f.close()

    db = DBSCAN(eps=1.03, min_samples=5).fit(features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    calculate_fscore_dbscan(db, df_processed)

    pca = PCA(n_components=2721)
    pca_vecs = pca.fit(features.toarray())
    pca_vecs = pca_vecs.transform(features.toarray())

    draw_PCA(db, pca_vecs)


# PCA number of components
def get_PCA_comp_num(features):
    pca = PCA()
    pca_vecs = pca.fit(features.toarray())

    plt.figure(figsize=(10, 2723))
    line, = plt.plot(pca_vecs.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.xlabel('Number of documents')
    plt.ylabel('Cumulative Explained Variance')
    ydata = line.get_ydata()
    index = np.where(ydata <= 0.90)
    plt.show()
    return index[0][-1]  # 1547


# PCA number of components vol2 - worse
def get_PCA_comp_num_vol2(features):
    pca = PCA(n_components=None)
    pca.fit(features.toarray())

    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    plt.bar(range(1, 2722), exp_var, align='center',
            label='Individual explained variance')

    plt.step(range(1, 2722), cum_exp_var, where='mid',
             label='Cumulative explained variance', color='red')

    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig("Barplot.png")
    plt.show()


# get kmeans number of clusters
def get_clust_numb_kmeans(pca_vecs):
    wcss = []
    for i in range(1, 40):
        kmeans_pca = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans_pca.fit(pca_vecs)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize=(10, 2723))
    plt.plot(wcss, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


# kmeans+pca
def kmeans_with_pca(df_processed, features):
    pca = PCA(n_components=1547)
    pca_vecs = pca.fit(features.toarray())
    pca_vecs = pca_vecs.transform(features.toarray())

    get_clust_numb_kmeans(pca_vecs)

    kmeans_pca = KMeans(n_clusters=cluster_number, init="k-means++", random_state=42)
    kmeans_pca.fit(pca_vecs)

    calculate_fscore_kmeans(kmeans_pca, df_processed, cluster_number)

    draw_PCA(kmeans_pca, pca_vecs, True, cluster_number)


# dbsacn+pca
def dbsacn_with_pca(df_processed, features):
    pca = PCA(n_components=1547)
    pca_vecs = pca.fit(features.toarray())
    pca_vecs = pca_vecs.transform(features.toarray())

    db_pca = DBSCAN(eps=1.03, min_samples=5).fit(pca_vecs)

    calculate_fscore_dbscan(db_pca, df_processed)

    draw_PCA(db_pca, pca_vecs, False, num=len(df_processed.groupby('cluster').groups))


def new_unique_words(df):
    most_common_word_count = 10
    string = ''
    products_list = df.values.tolist()
    for elem in products_list:
        for x in elem:
            string += x

    string = string.replace('][', ', ')
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace("'", '')
    string = string.replace("'", '')
    string = string.split(', ')
    return string


def unique_words(documents):
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo cywilne'])
    pc = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo administracyjne'])
    pa = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo farmaceutyczne'])
    pf = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo pracy'])
    ppr = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo medyczne'])
    pm = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo karne'])
    pk = new_unique_words(df)
    df = pd.read_csv('data/unique_result.csv', usecols=['prawo podatkowe'])
    ppo = new_unique_words(df)
    ultimate_list = pc + pa + pf + ppr + pm + pk + ppo
    f = open("boosteddata.csv", 'w')
    f.write('boosted \n')
    for sentence in documents:
        if isinstance(sentence, str):
            for word in ultimate_list:
                if word in sentence:
                    sentence += " " + str(word)
                    sentence += " " + str(word)
                    sentence += " " + str(word)
                    sentence += " " + str(word)
                    sentence += " " + str(word)
            f.write(sentence)
            f.write("\n")
    f.close()


def ultimate_loop_of_optimalization_kmeans(df_processed, documents, num1, num2, num3):
    stop_words_pl = pd.read_csv('https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt',
                                header=None)
    stop_words_pl = set(stop_words_pl[0].values)
    klient = open("klient.csv", "r")
    odmiany = klient.readlines()
    for odmiana in odmiany:
        lament = odmiana.split(',')
        odmiana = lament[0]
        stop_words_pl.add(odmiana)
    add_unique_words_to_documents(documents, num1, num2)
    boost = pd.read_csv('boosteddata.csv', encoding='iso-8859-2')
    boosted = pd.DataFrame(zip(boost['boosted'].tolist()), columns=['boost'])
    documents2 = boosted['boost'].values.astype("U")
    add_all_but_most_common_words_to_documents(documents2, num3)
    boost2 = pd.read_csv('boosteddata2.csv', encoding='iso-8859-2')
    boosted2 = pd.DataFrame(zip(boost2['boosted'].tolist()), columns=['boost'])
    documents2 = boosted2['boost'].values.astype("U")
    vectorizer = TfidfVectorizer(stop_words=list(stop_words_pl))
    features = vectorizer.fit_transform(documents2)
    kmeans_with_pca(df_processed, features)


def ultimate_loop_of_optimalization_dbscan(df_processed, documents, num1, num2, num3):
    stop_words_pl = pd.read_csv('https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt',
                                header=None)
    stop_words_pl = set(stop_words_pl[0].values)
    klient = open("klient.csv", "r")
    odmiany = klient.readlines()
    for odmiana in odmiany:
        lament = odmiana.split(',')
        odmiana = lament[0]
        stop_words_pl.add(odmiana)
    add_unique_words_to_documents(documents, num1, num2)
    boost = pd.read_csv('boosteddata.csv', encoding='iso-8859-2')
    boosted = pd.DataFrame(zip(boost['boosted'].tolist()), columns=['boost'])
    documents2 = boosted['boost'].values.astype("U")
    add_all_but_most_common_words_to_documents(documents2, num3)
    boost2 = pd.read_csv('boosteddata2.csv', encoding='iso-8859-2')
    boosted2 = pd.DataFrame(zip(boost2['boosted'].tolist()), columns=['boost'])
    documents2 = boosted2['boost'].values.astype("U")
    vectorizer = TfidfVectorizer(stop_words=list(stop_words_pl))
    features = vectorizer.fit_transform(documents2)
    dbsacn_with_pca(df_processed, features)


# ---------------------------------------------------------------------------------------------------------------------------#

df = pd.read_csv('dbdata.csv', encoding='utf-8')

other_column = []
other = ['odpowiedzi niestandardowe', 'prawo miädzynarodowe', 'tu interpolska', 'prawo konstytucyjne']
for index, row in df.iterrows():
    if row['label_high'] in other:
        other_column.append('inne')
    else:
        other_column.append(row['label_high'])

df['final'] = other_column

df_processed = pd.DataFrame(zip(df['text_full'].tolist(), df['final'].tolist()), columns=['sample', 'label'])
df_processed.head(5)
df_processed.shape

stop_words_pl = pd.read_csv('https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt',
                            header=None)
stop_words_pl = set(stop_words_pl[0].values)

klient = open("klient.csv", "r")
odmiany = klient.readlines()

for odmiana in odmiany:
    lament = odmiana.split(',')
    odmiana = lament[0]
    stop_words_pl.add(odmiana)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

documents = df_processed['sample'].values.astype("U")

vectorizer = TfidfVectorizer(stop_words=list(stop_words_pl))
features = vectorizer.fit_transform(documents)

# kmeans_with_pca(df_processed, features)
# dbsacn_with_pca(df_processed, features)

# ultimate_loop_of_optimalization_kmeans(df_processed, documents, 2, 12, 12)
# ultimate_loop_of_optimalization_dbscan(df_processed, documents, 6, 5, 11)
#

# kmeans_alg(df_processed, features)
# kmeans_with_pca(df_processed, features)
# dbscan_alg(df_processed, features)
# dbscan_eps_designation(df_processed, features)
# get_PCA_comp_num(features)
# kmeans_with_pca(df_processed, features)
# get_PCA_comp_num_vol2(features) #gorzej widać
# dbsacn_with_pca(df_processed, features)
# jaca_things()
# unique_words(df_processed['sample'])
