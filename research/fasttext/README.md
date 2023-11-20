Solution Description

Our solution is structured into three key components:
1. Input Preprocessing

   In this initial phase, we focus on refining the input data for efficient programming language identification.
   This involves the removal of extraneous information that does not contribute to language recognition.
   For instance, we replace each integer with a designated special token and eliminate non-ASCII characters.
   By doing so, we aim to keep our tokenizing dictionary as concise as possible. Additionally, during this step,
   we introduce special tokens for specific programming language structures, such as changing ":=" to <SPECIAL_QUAL>.
   This facilitates the model's ability to identify languages employing these distinctive structures.

3. Code/No-Code FastText Classification

   In this stage, we employ a FastText classification model to differentiate between code and non-code segments.
   This step is crucial for addressing class imbalance issues in programming language classification.
   If code is detected during this phase, we proceed to the next step. Otherwise, we classify the input as TGLANG_LANGUAGE_OTHER.
   The FastText classification model used here consists of just two classes.

5. Programming Language FastText Classification

   Following the code/no-code classification, we predict the programming language of the given code
   snippet using a separate FastText classification model. This final step completes the language
   identification process, ensuring accuracy and efficiency in classifying programming languages.
