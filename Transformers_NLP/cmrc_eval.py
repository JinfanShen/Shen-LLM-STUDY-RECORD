import sys
import os
import re

import nltk
tokenize = lambda x: nltk.word_tokenize(x)  #创建英文分词函数名

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    """对中英文混合文本进行分词处理"""
    in_str = str(in_str).lower().strip()  #转为字符串、小写、去除首尾空格
    segs_out = [] #存储分词结果
    temp_str = "" #临时存储连续的英文字符
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:  #如果是标点符号 且rm_punc为真
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:  # 判断是否是中文或标点符号
            if temp_str != "":  #有暂存的英文
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":  
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# remove punctuation
def remove_punctuation(in_str):
    """移除所有标点符号"""
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] #创造一个len(s2)+1列，len(s1)+1行的二维全0矩阵m
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax  #返回最长字串和它的长度

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        #分词并移除标点
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        
        #找到最长公共子串
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)

        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision     = 1.0*lcs_len/len(prediction_segs)
        recall         = 1.0*lcs_len/len(ans_segs)
        f1             = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)

def calc_em_score(answers, prediction):
    """计算精确匹配分数"""
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

# predictions: {example_id: prediction_text}
# references:  {example_id: [answer1, answer2, ...]}
def evaluate_cmrc(predictions, references):
    f1 = 0
    em = 0
    total_count = 0  #总问题数
    skip_count = 0   #跳过的问题数
    for query_id, answers in references.items():
        total_count += 1
        if query_id not in predictions:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue
        prediction = predictions[query_id]
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {
        'avg': (em_score + f1_score) * 0.5, 
        'f1': f1_score, 
        'em': em_score, 
        'total': total_count, 
        'skip': skip_count
    }

if __name__ == '__main__':
    pass