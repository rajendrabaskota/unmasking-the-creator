# **Unmasking The Creator**
### **This repository deals with classification of AI-generated and Human-crafted texts.**

---
## **Demo video for text classification**
https://github.com/rajendrabaskota/unmasking-the-creator/assets/66084649/75212559-6a60-4a5f-b71f-05c37cedbb89

---
## **Datasets Insights**
* ### **HC3-English dataset**
    * **Human-written texts - 24,300**
    * **AI-generated texts - 24,300**

* ### **GPT Wiki Intro dataset**
    * **Human-written texts - 150,000**
    * **AI-generated texts - 150,000**
---

## **Ongoing Tasks**
* ### **AI Dataset Generation using Llama-2-7B**
    * **Generated texts: 80,000**
    * **Goal: 200,000**
* ### **Model fine-tuining**
* ### **Exploration of different techniques**
---

## **Methodology**
The classification of text as AI-generated or human-crafted is based on 3 different methods:

* ### **Direct Text Classification using RoBERTa**
This method uses RoBERTa model which can directly classify whether the given text is written by an AI or a human. The model is fine-tuned with our above mentioned datasets.

* ### **Classification using Perplexity Method**
This method uses GPT-2 model to calculate perplexity of the given text. The calculated perplexity of the text is compared with the threshold value which is calculated from our dataset. The text having perplexity score lower than the threshold value is classified as AI-generated whereas the text is written by human if its perplexity is higher than the set threshold.

* ### **Domain-wise Perplexity Score Method**
The primary concept of this method is the same as the above method except that this method uses different thresholds for different texts based on the domain of the text. Thus, this method first classifies the given text into its respective domain using a fine-tuned RoBERTa model. The domains are based on the above mentioned datasets and the datasets are divided into 6 different domains: HC3 Reddit Eli5, HC3 Open QA, HC3 Medicine, HC3 Finance, HC3 Wiki CSAI and GPT-Wiki-Intro. After identifying the domain of the text, the perplexity of the given text is compared with the mean perplexity threshold of the respective domain. Finally, the text is classified as AI-generated or Human-crafted.
