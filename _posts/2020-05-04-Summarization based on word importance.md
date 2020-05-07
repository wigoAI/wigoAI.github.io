---
layout: post
title: "Summarization based on word importance"
tags: [Summarization]
use_math: true
comments: true
author: indexxlim
classes: wide

---

Improving the Estimation of Word Importance for News Multi-Document Summarization 논문을 참고하여 특징에 기반한 요약기를 실험해보려고 합니다.


# Word Importance
특징중에서 각 단어에 대한 특징을 사용하여 가장 중심문장을 판별합니다. 
논문에는 frequency feature, word type, KL-Divergence, NYT-weight, Unigram, Context feature, 그리고 MPQA and LIWC dictionary 등을 사용해서 문장의 중요도를 판답합니다.
이 때 regression을 이용해서 p-value로 각 특징으로써 검증을 평가하여 불필요한 특징을 고를 수 있습니다.
그러나 현재 몇몇 사전은 유료로 판매되고 있기 때문에 몇몇 특징만 사용하여 문장의 중요도를 나타내 보겠습니다.
각 현재 실험해볼 특징으로써
1. earliest first location 
2. latest last location
3. average location
4. average first location
5. appears in the first sentence
6. the number of times it appears in a first sentece 
7. LLR chi-square static value and MRW scores 

그리고 단어의 타입을 구분해주기 위해 품사인 POS(part-of-speech)와 개체명 인식 NER(Named Entity Recognition)도 추가로 특징으로 사용합니다.


# Propability and Frequency
위에서 나열한 특징들은 기본적으로 확률과 빈도에 기반한 특징들입니다.
[SumBasic]의 알고리즘은 다음과 같습니다.
Step 1 Compute the probability distribution over the words wi appearing in the input, p(wi) for every i; p(wi) = n N , where n is the number of times the word appeared in the input, and N is the total number of content word tokens in the input. 
Step 2 For each sentence Sj in the input, assign a weight equal to the average probability of the words in the sentence, i.e. weight(Sj ) = P wi∈Sj p(wi) |{wi|wi ∈ Sj}| 
Step 3 Pick the best scoring sentence that contains the highest probability word. 
Step 4 For each word wi in the sentence chosen at step 3, update their probability pnew(wi) = pold(wi) · pold(wi) 
Step 5 If the desired summary length has not been reached, go back to Step 2.

하지만 이 요약 방법은 [harry poter Summary]에서 보면 적절한 요약 문장을 선별하지 못하는 것으로 보입니다.
이런 확률에 기반한 요약 방법에 여러 특징을 추가함으로써 좋은 요약 결과를 추출해 보겠습니다.





# Regression-Based Keyword Extraction
회귀분석을 통해 각각의 특징이 얼마나 좋은 특징인지 판별해 보겠습니다.
회귀분석에 필요한 데이터셋은 CNN데이터를 이용하여 어떤 단어가 요약문장에 들어가있는지를 1,0로 라벨링해서 회귀를 실험해봤습니다.



<img src="/assets/summarization2/regression_result.png" itemprop="image" width="80%">



# Result
Rouge-N 은 문서 요약 분야에서 자주 이용되는 성능 평가 척도입니다.

이 특징에 기반하여 뽑은 문장과 DailyMail Dataset의 
여기서 Rouge-1, Rouge-2, Rouge-L로 평가해 봤습니다.

sumr_1 :  0.4543195393753824
sumr_2 :  0.13510762153701156
sumr_l :  0.34334198141421646


Lexrank Rouge점수와 비교해보면
sumr_1 :  0.43921486912181934
sumr_2 :  0.14790742692883918
sumr_l :  0.3266996620839612


Rouge 1-gram일 때와 전체문장으로 비교했을 때 비교적 높은 점수를 얻은 것으로 확인하였습니다.


#### -------------------------------------------------------------------------------  

추후 특징을 추가해서 실험 예정(pretraining Embedding)


[SumBasic]: https://www.cis.upenn.edu/~nenkova/papers/ipm.pdf
[harry poter Summary]: https://towardsdatascience.com/text-summarization-on-the-books-of-harry-potter-5e9f5bf8ca6c

[Text Summarization with Python]: https://medium.com/@umerfarooq_26378/text-summarization-in-python-76c0a41f0dc4
[Towards Automatic Text Summarization: Extractive Methods]: https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715