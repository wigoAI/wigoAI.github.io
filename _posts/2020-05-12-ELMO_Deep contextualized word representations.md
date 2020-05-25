---
layout: post
title: "ELMo-Embeddings from Language Model"
tags: [Attention, Transformer]
use_math: true
comments: true
author: indexxlim
classes: wide

---


# ELMo(Embeddings from Language Model)

이번 포스트는 2018년 상반기에 나왔던 ELMo에 대해 리뷰를 하고자 합니다.
ELMo는 세서미 스트리트(Sesame Street)라는 어린이용 인형극 애니메이션에서 나온 이름으로 Embeddings from Language Model의 약자입니다.
직역하면 언어모델부터 임베딩으로, 결국 언어모델을 사전 훈련(Pre-trained)했다는 이야기겠죠?

여기서 사전훈련 모델에 대해 잠깐 이야기하자면, 단순히 정확도만으로 생각한다면 무조건 Bert를 사용해야 된다고 생각 할 수 있습니다.  
하지만 Bert는 큰 결점이 있습니다. 기본 모델의 Layer의 갯수가 12, 히든 레이어의 차원수가 768로 너무 크기 때문에 반드시 고사양 환경이 필수적이라는 것입니다.   
Bert Base와 GPT-2 Base 모델들의 파라미터 갯수는 대략 110milion이고 ELMo의 Small모델은 13milion, Medium모델은 28milion으로 많은 차이가 나기 때문에 비교적 장점이 될 수 있습니다. ~~ALBert와 Bert-small은?~~  
그래서 사용자의 관점에서 CBOW, Skip-Gram 같은 모델부터 ELMo, GPT-2, 그리고 Bert 까지 환경을 고려해서 차선택을 골라 합리적인 모델링을 할 수 있습니다. 

여기까지 모델링의 관점에서 차선택을 위해서 알고 있어야 한다는 관점이었고,  
개인적으로는 ALbert와 Bert-small 등의 모델의 결과와 비교하기 전에 ELMo의 이름이 인상적이었습니다.  
원론적으로 **ELMo - Embeddings from Laguage Model**이라는 개념은 대대수의 모델링과 같은 개념이라고 생각되는데요.
언어모델은 각 단어가 등장할 확률을 할당(assign)한 상태공간(state space)이고, n-gram 이상이 적용 되면 그 상태(state)는 단어 확률 변수의 마르코프 연쇄로 이루어집니다.
그 예로 은닉 **마르코프 모형(hidden Markov model, HMM)**에서는 은닉 상태(hidden state)와 관찰가능한 결과를 통해 상태확률 혹은 우도(likelihood)를 계산합니다.
여기서 신경망까지 나간다면, [SVDL]에서는 **interpretation of a simple RNN as the maximum likelihood estimate of a state space model that follows deterministic dynamics** 라고 합니다.
그래서 신경망과, 언어모델, 그리고 임베딩까지, ELMo의 구성요소를 알아보겠습니다.


## 1. LM

ELMo는 LM에 기반하여 학습을 진행합니다.  
통계학적으로 Language Model(LM)이란 일종의 확률분포(pobability distribution)입니다. 언어모델은 단어가 m개가 주어졌을때 각 단어의 확률 P(w1 ... wm)을 가지고 있습니다.
언어모델은 대다수의 자연어의 모델에 사용되는데, 이미지가 어떤 필터(filter)가 포함되어 있는지를 특징으로 학습한다면, 자연어는 각 단어의 확률분포로부터 단어 조합의 확률분포를 학습한다고 의의를 둘 수 있습니다.
n-gram개의 단어로 이루어진 문장의 확률을 계산한다면 마르코프 체인 법칙(Markov chain rule)을 통해 계산 할 수 있습니다. 
이 최대 우도 추정(maximum likelihood estimation) 수학적인 도출은 ∏i=1nP(wi) 

[N-gram models]은 Smoothing 이나 Adjusting count로 변형 될수도 있습니다.  
MLE unigram probabilities 은 다음과 같을 때,   $P(wx) = count(wx) \over N$  
$N$은 $∑w count(w)$이고 $V$는 $vocab size$ 라고 했을 때, Add-one smoothing은 다음과 같은 절차를 지납니다.  
Smoothed unigram probabilities  $P(w_x) = count(w_x) \over (N+V)$  
Adjusted counts (unigrams)      $c_i^* = (c_i+1) {N \over (N+V)}$  
MLE bigram probabilities $P(w_n\|w_{n-1}) = {count(w_{n-1}w_n ) \over count(w_{n-1})}$  
Laplacian bigram probabilities $P(w_n\|w_{n-1}) = {count(w_{n-1}w_n )+1 \over count(w_{n-1})+V}$

 
직접적으로 Add-one smoothed bigram probabilites을 살펴보면 다음과 같은 테이블이 나옵니다.

<img src="/assets/elmo/original_count.png" itemprop="image" width="100%"> | <img src="/assets/elmo/new_count.png" itemprop="image" width="100%"> |

<img src="/assets/elmo/original_probabilites.png" itemprop="image" width="100%"> | <img src="/assets/elmo/new_probabilites.png" itemprop="image" width="100%">

즉 주어진 문장이 N개의 token으로 이루어져 있을때, $(t_1,t_2,...t_N)$ forward language model은 $(t_1,t_2,...,t_k)$에 대해 $t_k$의 확률을 모델링함으로써 문장의 확률을 구합니다.
$p(t_1,t_2,...t_N)=∑k=1Np(t_k\|t_1,t_2,...,t_{k−1})$

최근 좋은 성능을 보여주는 언어 모델들은 Bi-LSTM을 통해 문맥과는 독립적인 token 표현 xLMk을 만들어낸다. 그후 forward LSTM을 통해 L개의 layer를 통과시킨다. 각 token의 위치 k에서, 각 LSTM layer는 문맥에 의존되는 표현 $h→LMk,L$을 생성시킨다. 이 때,  $j=1,...,L$이다. LSTM layer의 가장 최상위층 output인 $h→LMk,L$은 softmax layer를 통해 다음 토큰인 $t_k+1$을 예측하는 데에 사용한다.

backward LM 역시 forward LM과 비슷한데, 그 차이점은 문자열을 거꾸로 돌려 작동한다는 점에 있다. 즉 현재 token보다 미래에 나오는 token들을 통해 이전 token들을 예측하는 메커니즘이다.

$p(t_1,t_2,...,t_N)=∑k=1Np(t_k\|t_{k+1},t_{k+2},...,t_N)$


이것은 또한 주어진 $(t_{k+1},...,t_N)$에 대하여 $t_k$의 표현인 $h←LMk,j$를 예측한다.

BiLM은 forward LM과 backward LM을 모두 결합한다. 다음의 식은 정방향 / 역방향의 log likelihood function을 최대화시킨다:

$∑k=1N(logp(t_k\|t_1,...,t_{k−1};θx,θ→LSTM,θs)+logp(t_k|t_{k+1},...,t_N;θx,θ←LSTM,θs))$


여기서 token 표현 θx와 softmax layer θs의 파라미터를 묶었으며, 각 방향의 LSTM의 파라미터들은 분리시킨 채로 유지하였다. 여기서 이전 연구들과 다른 점은 파라미터들을 완전히 분리하는 것 대신에 방향 사이에 일정 weight를 공유하도록 하였다.

## 2. ELMo

"deep contextualized" 의 새로운 타입을 소개한다.  
ELMo는 (1) 단어 사용의 복잡한 특성(Syntatic and Semantic, 구문과 의미론)을 모두 내포하며 (2) 이러한 용도가 언어적 맥락(다의어나 동음이의어)에 따라 어떻게 달라지는지 표현(word representation) 할수 있는 사전학습(pre-training) 모델이다. 단어 벡터는 대형 텍스트 말뭉치를 통해 양방향 언어 모델(biLM)로 내부적으로 학습했다. 이러한 벡터들이 기존 모델들에 쉽게 추가될 수 있고 질문 답변, 문자 첨부, 감정 분석 등 6가지 NLP 문제에서 개선할 수 있었다.   
ELMo의 representations은 전체 입력문장을 사용하고, 양방향 LSTM을 통해 유도된 벡터는 대형 말뭉치에 기반한 언어모델을 목표로 학습된다. 구체적으로는 최상위 레이어(top LSTM layer)만 사용하던 기존과 달리 쌓인(stacked) 입력 데이터를 모두 결합해서 사용하기 때문에 성능이 좋다.  
저자는 [Intrinsic evaluation]을 통해서 higher, lower level 특징을 파악할 수 있었는데, Intrinsic evaluation은 만들어진 임베딩 벡터를 평가할 때 Word analogy와 같은 Subtask를 구축하여 임베딩의 품질을 특정 중간 작업에서 평가하는 방식이다.  Real task의 정확도를 통해서 임베딩의 품질을 평가하는 것이 아기 때문에 유용성을 판단하기 위해 실제 작업과의 긍정적인 상관관계 분석이 필요하다.  
higher-level LSTM : 문맥을 반영한 단어의 의미를 잘 표현함  
lower-level LSTM : 단어의 문법적인 측면을 잘 표현함  



#### Bidirectional language models

ELMo는 biLM에서 등장하는 중간 매체 layer의 표현들을 특별하게 합친 것을 의미한다. 각 토큰 tk에 대하여, L개의 layer인 BiLM은 2L+1개의 표현을 계산한다.

Rk={xLMk,h→LMk,j,h←LMk,j|j=1,...,L}

Rk={hLMk,j|j=0,...,L}


이 때, hLMk,0은 token layer를 뜻하고, hLMk,j=[h→LMk,j;h←LMk,j]는 biLSTM layer를 의미한다. 그래서 모든 layer에 존재하는 representation을 R로 single vector로 혼합하는 과정을 거친다:
ELMok=E(Rk;θe)

예시로, 가장 간단한 ELMo 버전은 가장 높은 layer를 취하는 방법이 있다: E(Rk)=hLMk,L.

이 ELMo는 task에 맞게 또 변형될 수 있다.

ELMotaskk=E(Rk;θtask)=γtask∑j=0LstaskjhLMk,j.


stask는 softmax-normalized weight를 의미하고 γtask는 task model을 전체 ELMo vector의 크기를 조절하는 역할을 맡는다.

어떤 의미인지 그림을 통해 살펴보도록 하자.



해당 그림과 같이 각 Bi-LSTM Layer들을 통해 나오는 hidden representation을 task의 비율에 맞추어 더해 ELMo vector를 만드는 것이다.

즉 과정을 다시 정리해보면
(1) BiLSTM layer의 꼭대기층의 token이 softmax layer를 통해 다음 token을 예측하도록 훈련시킨다.
(2) 훈련된 BiLSTM layer에 input sentence를 넣고 각 layer의 representation 합을 가중치를 통해 합한다.
(3) input sentence length만큼 single vector가 생성된다.

3) Using biLMs for supervised NLP tasks
이렇게 pre-trained된 biLM과 NLP task를 위한 supervised architecture를 결합하여 task model의 성능을 향상시켰다. 이 때, 이 논문에서는 이 representation의 선형 결합을 다음과 같이 학습시켰다.

(1) 먼저 biLM없이 supervised model의 가장 낮은 layer를 고려했다. 대부분의 supervised NLP model은 가장 낮은 층에서 공통적인 architecture를 공유한다. 따라서 이는 ELMo를 쉽게 붙일 수 있는 계기가 되었다. 주어진 sequence (t1,...tN)에 대해 pre-trained token representation인 xk를 사용했으며 때때로 문자 기반 representation을 사용하는 것이 NLP task의 대부분이였다. 이 때, bi-RNN이나 CNN, feed forward network를 통해 context에 의존적인 representation hk를 만드는 것이 NLP task에서 주로 하는 작업이였다.

(2) 따라서 supervised model에 ELMo를 부착하기 위해 biLM의 가중치값을 고정시키고 ELMo vector인 ELMotaskk를 token representation xk와 결합시켜 다음과 같은 [xk;ELMotaskk] representation을 생성하였다. 이 representation은 context-sensitive representation hk를 만들기 위한 input으로 사용된다.

(3) 마지막으로 ELMo에 dropout을 설정하는 것이 더 좋다는 것을 알았으며, 때때로 λ||w||22와 같은 regularization factor를 더하는 게 좋아 몇몇 케이스에서는 regularization을 생성하였다.

## 5. 결론

본 연구에서는, 전적으로 주의를 기반으로 한 최초의 시퀀스 전달 모델인 Transformer를 제시하여, 인코더-디코더 아키텍처에서 가장 일반적으로 사용되는 순환 레이어를  multi-headed self-attention로 대체하였습니다.  
번역 작업의 경우, Transformer는 순환 또는 합성곱  레이어에 기반한 구조보다 훨씬 더 빠르게 훈련될 수 있습니다. WMT 2014 영어-독일어 및 WMT 2014 영어-프랑스어 번역 과제 모두에서 SOTA를 달성했습니다. 이전의 과제에서 우리의 최고의 모델이 이전에 보고된 모든 앙상블보다 더 성능이 좋습니다. 우리는 관심 기반 모델의 미래에 대해 흥분하고 있으며 다른 과제에 적용할 계획입니다. Transformer를 텍스트 이외의 입력 및 출력으로 확장하고 영상, 오디오, 비디오 등에서 대용량 입력과 출력을 효율적으로 처리하기 위한 국부적이고 제한된 어탠션 메커니즘을 조사할 계획입니다. 우리의 또 다른 연구 목표는 덜 순차적으로 발전하는 것입니다.

## Reference

[N-gram models]: http://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf
[Statistical Parametric Speech Synthesis]: https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/44312.pdf
[Intrinsic evaluation]: https://cs224d.stanford.edu/lecture_notes/notes2.pdf
[SVDL]: http://blog.shakirm.com/wp-content/uploads/2015/07/SVDL.pdf