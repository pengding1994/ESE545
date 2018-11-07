import numpy as np
import random
import csv
import time
from matplotlib import pyplot as plt
from itertools import combinations    
    
def problem1():
    global ct
    global user_number
    ct = 0
    user_number = 0
    dic = {}
    with open('Netflix_data.txt', 'r') as f:
        user_abandon = set()
        for line in f:
            if ':' in line:
                ct += 1
                mnum = line[:-2]
                mnum = int(mnum)
            else:
                curr = line.strip().split(',')
                id = curr[0]
                rate = curr[1]
                id = int(id)
                rate = int(rate)
                if rate >= 3:
                    cur_user_movie = dic.get(id,[])
                    if cur_user_movie == str('NA'):
                        user_abandon.add(id)
                    else:
                        dic[id] = cur_user_movie + [mnum]
                        if len(cur_user_movie)>=20:
                            dic[id] = str('NA')
                            user_abandon.add(id)
    for userid in user_abandon:
        if userid in dic.keys():
            del(dic[userid])
    user_number = len(dic)
    matrix = np.zeros((ct,user_number),dtype=np.int32)
    user_1 = 0    
    for key in dic.keys():
        cur = dic.get(key)
        for m_number in cur:
            matrix[m_number-1][user_1] = 1
        user_1+=1
    return(matrix)
    
def problem2(matrix):
    user_number = len(matrix[0])
    user_random = random.sample(range(0,user_number),20000)
    dis = []    #distance of 10000 pairs
    for i in range(0,10000):
        user1 = matrix[:,user_random[i]]
        user2 = matrix[:,user_random[19999-i]]
        numerator = np.sum(np.bitwise_and(user1,user2))
        denominator = np.sum(np.bitwise_or(user1,user2))
        jac_dis = 1-int(numerator)/int(denominator)
        dis.append(jac_dis)
    dis_min = np.min(dis)
    dis_avg = np.average(dis)
    print("Average distance is: ", str(dis_avg))
    print("Lowest distance is: ", str(dis_min))
    bins = 20
    plt.hist(dis, bins=bins, alpha=0.5)
    plt.title('Jaccard Distance Histogram')
    plt.xlabel('Jaccard Distance')
    plt.ylabel('Count')
    plt.show()

def problem3(matrix):
    dic = {}
    for i in range(user_number):
        one_index = np.flatnonzero(matrix[:,i])
        dic[i] = one_index
    return (dic)

def problem4(dic,minhash_size):    
    R = 4507
    a = random.sample(range(1, ct),minhash_size*2)
    b = a[0:minhash_size]
    a = a[minhash_size:minhash_size*2]
    sig_matrix = np.zeros((minhash_size,user_number),dtype=np.int32)
    #generate signature matrix
    for i in range(minhash_size):
        h = []
        for x in range(ct):
            h.append((a[i]*x+b[i])%R)
        h = np.asarray(h)
        for user in range(user_number):
            ones_index = dic.get(user)
            hash_val = min(h[ones_index])
            sig_matrix[i,user] = hash_val          
    #generate LSH
    lsh_hashtables,lsh_a,lsh_b,lsh_P = get_lsh_hashtable(sig_matrix,50)
    #get candidate pairs
    candidate_pair = elect_candidates(lsh_hashtables)
    #get close pairs
    similar_pairs = get_closeusers_indices(candidate_pair,sig_matrix,0.35)
    #write to csv file
    with open('similarPairs.csv','w') as writeFile:
        similarWriter = csv.writer(writeFile, delimiter=',')
        for i in range(len(similar_pairs)):
            similarWriter.writerow([similar_pairs[i][0], similar_pairs[i][1]])
    return sig_matrix,a,b,R,lsh_hashtables,lsh_a,lsh_b,lsh_P,candidate_pair,similar_pairs  

def problem5(query_vec,sig_matrix,a,b,P,lsh_hashtables,lsh_a,lsh_b,lsh_P):
    inputvec =np.asarray(query_vec)
    #inputsparsevec=np.flatnonzero(inputvec)
    minhash_vec = get_minhash_vec(inputvec,a,b,P,np.shape(sig_matrix)[0])
    #inputvec_hashvalue = str(lsh_hash(minhash_vec,a,b,P))
    candidates = query_candidates(minhash_vec,lsh_hashtables,lsh_a,lsh_b,lsh_P)
    #candidates1 = elect_candidates(sig_matrix[:,0],lsh_hashtables,lsh_a,lsh_b,lsh_P)
    close_users_indices =query_close_user(minhash_vec,candidates,sig_matrix,0.35)
    return close_users_indices

def query_candidates(minhash_vec,lsh_hashtables,lsh_a,lsh_b,lsh_P):
    candidates=[]
    band_num = len(lsh_hashtables)
    unitrows =len(minhash_vec)/band_num
    unitrows =int(unitrows)
    for band,hashtable in enumerate(lsh_hashtables):
        hashvalue = lsh_hash(minhash_vec[band*unitrows:(1+band)*unitrows],lsh_a[band],lsh_b[band],lsh_P)
        hashvalue = str(hashvalue)
        candidates +=hashtable.get(hashvalue,[])
    return np.unique((np.asarray(candidates)))

def query_close_user(minhash_vec,candidates,sig_matrix,threshold_score):
    if len(candidates)==0:
        return np.asarray([])
    threshold_num = float(len(minhash_vec) *2 *(1-threshold_score))/(2-threshold_score)
    tempmat = (np.sum(minhash_vec == (sig_matrix.transpose())[candidates], axis=1)) >= threshold_num
    temparray = np.where(tempmat)[0]
    if len(temparray)==0:
        return np.asarray([])
    close_users_indices=candidates[temparray]
    return close_users_indices

def get_minhash_vec(inputvec,a,b,P,minhash_size):
    inputsparsevec = np.flatnonzero(inputvec)
    vecsize =np.size(inputvec)
    minhash_vec =[]
    for i in range(minhash_size):
        h = []
        for x in range(vecsize):
            h.append((a[i]*x+b[i])%P)
        h = np.asarray(h)
        hash_val = min(h[inputsparsevec])
        minhash_vec.append(hash_val)
    return np.asarray(minhash_vec)
    
def get_closeusers_indices(candidate_pair,sig_matrix,threshold_score):
    threshold_num = 500*threshold_score
    similar_pairs = []
    sig_matrxT = sig_matrix.transpose()
    for pair in candidate_pair:
        user1 = pair[0]
        user2 = pair[1]
        if(np.sum(sig_matrxT[user1]==sig_matrxT[user2])>=threshold_num):
            similar_pairs.append([user1,user2])
    return similar_pairs

def elect_candidates(lsh_hashtables):
    candidates=[]
    for band,hashtable in enumerate(lsh_hashtables):
        for k,v in hashtable.items():
            if(len(v)>1):
                cur_candidates = list(combinations(v, 2))
                candidates.extend(cur_candidates)
    return list(set(candidates))

def lsh_hash(mat,lsh_a,lsh_b,lsh_P):
    return (lsh_a*mat+lsh_b)%lsh_P

def get_lsh_hashtable(sig_matrix,band_num):
    if np.shape(sig_matrix)[0]%band_num != 0:
        print(" not divisible band_num")
        return
    lsh_hashtables=[]
    unitrows =np.shape(sig_matrix)[0]/band_num
    unitrows = int(unitrows)
    lsh_P = 4507
    lsh_a = random.sample(range(1, ct), band_num*2)
    lsh_b = lsh_a[0:band_num]
    lsh_a = lsh_a[band_num:band_num*2]
    for group in range(0,band_num):
        lshdic ={}
        tempmat = sig_matrix[group*unitrows:(group+1)*unitrows]
        tempmat = lsh_hash(tempmat,lsh_a[group],lsh_b[group],lsh_P).transpose()
        for index, vec in enumerate(tempmat):
            vecstr = str(vec)
            hashunit = lshdic.get(vecstr,[])
            hashunit.append(index)
            lshdic[vecstr]=hashunit
        lsh_hashtables.append(lshdic)
        print(str(group) + 'group finished')
    return lsh_hashtables,lsh_a,lsh_b,lsh_P

if __name__ == '__main__':
    start = time.time()
    matrix = problem1()
    problem2(matrix)
    dic = problem3(matrix)
    sig_matrix,a,b,P,lsh_hashtables,lsh_a,lsh_b,lsh_P,candidate_pair,similar_pairs = problem4(dic,500)
    query_vec =(matrix[:,1]).transpose()
    close_user = problem5(query_vec,sig_matrix,a,b,P,lsh_hashtables,lsh_a,lsh_b,lsh_P)
    end = time.time()
    print(str(end-start)+'seconds to finish Project 1')