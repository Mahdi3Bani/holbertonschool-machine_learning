# Question Answering with BERT

This README file describes the tasks and scripts involved in creating a question-answering bot using BERT. The tasks demonstrate how to use TensorFlow Hub's BERT model for question answering, handle user input in a loop, perform semantic search, and extend the question-answering functionality to multiple reference documents.
## Prerequisites

Ensure you have the following libraries installed:

1-    tensorflow
2-    tensorflow_hub
3-    transformers
4-    numpy

You can install them using pip:

```bash
pip install tensorflow tensorflow_hub transformers numpy
```

# Task Descriptions
## Task 0: Question Answering

## Function: question_answer

This function finds a snippet of text within a reference document to answer a question using BERT.

## Requirements:

    question is a string containing the question to answer.
    reference is a string containing the reference document from which to find the answer.
    Returns a string containing the answer. If no answer is found, returns None.

## Usage:

    run the jupyter cells

## Example:

```python

question_answer('When are PLDs?', reference)
```

Expected Output:


on-site days from 9:00 am to 3:00 pm

## Task 1: Create the Loop

Create a script that takes input from the user with the prompt Q: and prints A: as a response. If the user inputs exit, quit, goodbye, or bye, print A: Goodbye and exit.

## Usage:

    run the jupyter cells

## Example:

```plaintext

Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
```

## Task 2: Answer Questions


This function answers questions from a reference text.

## Requirements:

    reference is the reference text.
    If the answer cannot be found in the reference text, respond with Sorry, I do not understand your question.

## Usage:

    run the jupyter cells

## Example

```plaintext

Q: When are PLDs?
A: on-site days from 9:00 am to 3:00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
```


## Task 3: Semantic Search


This function performs semantic search on a corpus of documents.

## Requirements:

    corpus_path is the path to the corpus of reference documents.
    sentence is the sentence from which to perform semantic search.
    Returns the reference text of the document most similar to sentence.

## Usage:

    run the jupyter cells

## Example:

```python

semantic_search('ZendeskArticles', 'When are PLDs?')
```

## Expected Output:

```plaintext

PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career.
```

## Task 4: Multi-reference Question Answering


This function answers questions from multiple reference texts.

## Requirements:

    corpus_path is the path to the corpus of reference documents.

## Usage:

    run the jupyter cells


## Example Interaction:

```plaintext

Q: When are PLDs?
A: on-site days from 9:00 am to 3:00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
```

## About BERT:

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model developed by Google. It is designed to understand the context of a word in search queries and text by looking at the words that come before and after it. This bidirectional approach allows BERT to achieve state-of-the-art results on a variety of natural language processing tasks.
Key Features of BERT:

    Bidirectional: BERT reads the entire sequence of words at once, allowing it to understand the context of each word more deeply.
    Pre-trained: BERT is pre-trained on a large corpus of text, which helps it perform well on a wide range of tasks with minimal fine-tuning.
    Transformer Architecture: BERT uses transformers, which are neural network architectures designed to handle sequential data and capture long-range dependencies.

## Applications of BERT:

    Question Answering
    Text Classification
    Named Entity Recognition (NER)
    Semantic Search
    Text Summarization

BERT has significantly advanced the field of natural language processing by providing a robust and flexible model that can be adapted to many different tasks with minimal effort.