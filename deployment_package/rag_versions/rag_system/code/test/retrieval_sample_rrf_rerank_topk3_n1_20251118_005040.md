# Retrieval qualitative sample (RRF BM25+Vector + Cross-Encoder Rerank)

- Timestamp: 20251118_005040
- Questions source: ground_truth_answers.csv (Answer Type == 'Answer - Reviewed')
- Retrieve top-10, rerank, return top-3
- Reranker: BAAI/bge-reranker-base
- Printed passage truncate (for this report only): 800 characters

## Q1. What is the best approach to communicate our loved one's MCI or dementia diagnosis to the rest of the family?
- ID: AA033, Category: Diagnosis
- Ground truth (reviewed):

When sharing a loved one's diagnosis of MCI (Mild Cognitive Impairment) or dementia with the rest of the family, it's important to do it in a considerate and understanding manner. Choose a quiet, comfortable setting where everyone can gather, then explain the situation clearly and honestly, using simple terms to describe the condition, its symptoms, the results of the diagnosis, and the expected progression. Be prepared to answer questions and provide resources for further information. Try to stay positive and emphasize that while this diagnosis is challenging, there are treatments and strategies to manage the symptoms. Remind them that they are not alone, and the integrity of the support system is crucial. Also, leave room for them to express their feelings and concerns, and offer reassur

- Retrieved passages:

[1] Once you share the diagnosis, explain what your loved one can still do and how much
he or she understands. You might offer suggestions for interacting, such as by having
people briefly reintroduce themselves and avoiding correcting your loved one if he or
she forgets something. Encourage people to engage in activities that are familiar to your
loved one.
A young child might look to your example to know how to act around a person who has
Alzheimer's. Show that it's OK to talk to your loved one and enjoy normal activities with
him or her, such as listening to music or reading stories. Older children might have a
harder time accepting the changes Alzheimer's can cause and might feel uncomfortable
spending time with a loved one who has Alzheimer's. Avoid forcing the issue. Instead,
talk honest
Source: /Users/gengminjie/Desktop/code/RAG/deployment_package/rag_versions/rag_system/code/downloaded_files/Disease/Diagnosis Sharing/How to share the disease with patients, friends and family_/VA.gov.pdf

[2] The journey toward a diagnosis of MCI is most often initiated by a person's subjective complaint
about memory and thinking problems or by concerns expressed by those closest to the
individual. This is an important distinction because cognitive changes that are readily apparent to
the broader outside world are more likely to signal that the person is suffering from dementia. In
addition, people who have progressed into Alzheimer's disease are often not aware of their
memory lapses.
You might also be interested in...
Cognitive Fitness Online Course
Wondering if you can affect your brain health, memory, and cognitive function? Learn in our
course about simple lifestyle changes you can make to optimize your cognitive fitness—and stay
mentally sharp!
Learn More!
Your regular doctor is a good pl
Source: /Users/gengminjie/Desktop/code/RAG/deployment_package/rag_versions/rag_system/code/downloaded_files/Disease/MCI/How is it diagnosed_ Cognitive testing, blood testing, what they should look for_/Harvard.health.pdf

[3] Helping Family and Friends
Understand Alzheimer's Disease
Español
When you learn that someone close to you has Alzheimer's
disease, deciding when and how to tell your family and
friends may be difficult. You may be worried about how
others will react to or treat your loved one. It's okay to wait
until you feel emotionally ready to share the news or to only
tell your closest family members and friends. By knowing
what is happening, the people you trust the most can help
support you and the person with Alzheimer's. The following
suggestions can help get you started.
Sharing the diagnosis
It may be hard to share a loved one's Alzheimer's diagnosis
with others. Here are a few suggested approaches:
Source: /Users/gengminjie/Desktop/code/RAG/deployment_package/rag_versions/rag_system/code/downloaded_files/Disease/Diagnosis Sharing/How to share the disease with patients, friends and family_/NIA.NIH.gov.pdf
