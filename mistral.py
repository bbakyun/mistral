from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 디렉터리 경로
model_name_or_path = "./Mistral-7B-Instruct-v0.2-GPTQ"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

# 각 문단 설정
paragraph1 = "We are going to talk about some of the data access challenges in We are also going to talk generation or RAG with Amazon Bedrock. We are going to then implement in our RAG application using We are then going to talk about some of the architecture and we are going to finish So what are some of the that we see in our We need to have a robust and once the user is to make sure the user only and not an uncontrolled data access. After that, we also want to make sure our system is scalable. So as we onboard more and it should be able to handle it instead of needing the application to be changed. After that, we also wanna make sure, because we are adding new we don't want to end up with data silos and we should still be able access to our data. And last but not the least, we don't want to impact our existing user experience. All of these stringent security integrate with our And all of that is encapsulated in what we call as data governance."
paragraph2 = "Let's start with RAG or So what is RAG? RAG is a mechanism with which we use our existing enterprise data and feed it to our Large to make it more context aware more accurate responses in Chatbot for example. Now, let's see RAG in action."
paragraph3 = "RAG has two main parts. So first of the parts is the data ingestion workflow. This is where we are for our application. So we have our data sources. As a company, you must have scattered everywhere in So we use that data, we as chunks, and these chunks are then sent to an embedding model, an embedding Large language model, which converts these and stores them in Vector So this is the data ingestion workflow. Now moving on to the next part. We have our text generation workflow. So this is where we have a user using our Chatbot application or and asking questions and So what happens is, the user asks question in natural language, which again gets sent to the embedding model, the question gets and then we do a search So we try to find similar documents. Once we retrieve those similar documents, we augment our prompt with what we retrieved in the search result, and then the augmented prompt is sent to a Large language model, which in turn creates the final response for our application. This is great, but imagine that we see here and the in natural language to embeddings and the whole creation of our for you by an AWS service."
paragraph4 ="""This is exactly what Knowledge Bases for Amazon Bedrock does for you. Amazon Bedrock, as you may to create, build, scale your in AWS using a single API. And you have a choice of So this is what Knowledge Bases for Amazon Bedrock does for you. It provides a single API RAG workflow for you. Behind the scenes, it's still that we would be creating Now in this diagram, the if we think about the AWS your data stored in an S3 bucket. And then the vector it could be an Amazon OpenSearch service. There are other choices with Knowledge Bases like Redis, Pinecone, Amazon Aurora, PG Vector, and actually more, so you But for this lightning Amazon OpenSearch service. Now let's see how with our application. So imagine a scenario of your So you have a lot of data and you have, for this example, of marketing data, which can be seen as the leads document here. And you also have a lot of financial data, which can be seen as the profit here in this diagram. So you have created the Vector documents, and then you have your organization. So you have CFO, CEO So you also have built let's say for your organization, and now the CFO is So he's asking questions like, "Give me a report of profit So the question that he asked gets sent to the Knowledge Bases for Amazon Bedrock, it does the search against and then based on what we send it to the Large and it creates a final response for us. So this is great. Now, if the same CFO is or marketing-related document, the same workflow will get triggered and he would still be able that he's asking from marketing documents. This is because the scope of the search is the entire Vector database. When we do the search, we search the entire vector Now what if we don't any marketing related data?" """
paragraph5 = "Persona-based access and let's see how we can do that. So again, we have our documents. So here you will see two which are our metadata files. So in addition to our marketing and finance related to upload some metadata files here. Now we are also going to integrate, so you may have some It could be Okta or any other So you're going to with your generative AI application. We have used Amazon but it could be anything. The request the CFO is asking The request gets passed to an which retrieves a filter. This filter is retrieved based on what access the CFO has; This filter along with the question that the CFO asks, gets for Amazon Bedrock, and a on the documents which match that filter. Now, whenever the filter that we originally used in the search result is retrieved. So because the CFO has access to all the finance related documents, the filter matches the metadata and only the relevant which are sent to the Large language model and a response is created. Now when the CFO is asking lead related stuff, the filter Now the documents are not returned. So that's how we prevent of the marketing data that So this is how we reduce the to only the filter criteria. So this was a quick walkthrough of how we implement persona-based access to our data"
paragraph6 = "So we looked at the RAG takes to build such an Now let's look at some of How would you go ahead and implement this? We have three patents. The first one here is a way to manually update this metadata. So let me walk you through that. You have a use case where you want to create a Chatbot application, you have a lot of personas, and now what you do is you you may have uploaded In this pattern, what you the metadata file along with the objects you have in your S3 bucket. So each document in your S3 would have a corresponding metadata file. Now, when the user signs in again, we are taking example the user, the application based on user or the role When Amazon Bedrock these documents along with to Amazon OpenSearch Vector store. So when the user prompt comes the filters are mapped to the and that's how the user only that they should have access to and not the entire Vector store. So this is similar to what The next pattern here is to use the default So when you create a Knowledge it automatically propagates which is nothing but the S3 prefix. So let's talk about the use case. Again, you have a lot of and you have a lot of but now you have these documents stored in each user's prefix, right? And prefixes are nothing synonymous to what we have in Amazon S3. We call prefix in S3. So each user has its own prefix. You upload documents in and Knowledge Bases would to the Vector store. So what happens is now when identity provider, the filter is added to the prompt or the query and this maps to the user or the role. And that's how the application understands what is the prefix that it With that information, to the metadata and that user only has access the corresponding documents The last pattern we have here is where, as an organization, you use This could be Microsoft could be Okta, it could be anything else, paying jump clouds, so on and so forth. In this pattern, similar the metadata is now propagated Now, S3 access grants allow you to have temporary credentials not just with IAM users and roles. So IAM identities, but With S3 access grant, these to the Amazon OpenSearch Vector And then when the user sign in using one of the identity provider, And with Access grants, you you get back temporary credentials, and then the application tier adds filter. The filter is mapped to the metadata and the user has access to and they're restricted to those documents and not have access to So these are some of the use There are other use cases and other patterns that you Now, we'll quickly see this in action."

# 문단 리스트
paragraphs = [paragraph1, paragraph2, paragraph3, paragraph4, paragraph5, paragraph6]

# Reflexion Framework 적용
iterations = 1  # Reflexion 반복 횟수

for i, paragraph in enumerate(paragraphs):
    current_paragraph = paragraph  # 초기 입력

    for iteration in range(iterations):
        try:
            print(f"Processing Paragraph {i + 1}, Iteration {iteration + 1}...")

            # 프롬프트 작성
            prompt = f"""
            Please read the following text and generate an explanation suitable for a **basic-level audience**. Use **friendly and everyday language** to make the explanation simple and relatable. Ensure that the output is written at a **Flesch-Kincaid Grade Level of approximately 6**. Follow the **exact format provided below**, and avoid including any information outside of this structure.

            ### Explanation Format:
            - **Explanation of the Topic**: 
              Provide a clear summary of the main topic or concept discussed in the input text. Use friendly and conversational language.

            - **IT Domain Terms and Definitions**: 
              Identify IT-related technical terms in the input text and explain each term in a way that's easy to understand. Use analogies or examples from daily life to make the explanation relatable. For example:
                - *"Cloud Computing"*:
                  Imagine storing your favorite photos and videos in a magical online photo album that you can access from any device, anywhere. That’s what cloud computing does for your data.

            ### Output Requirements:
            - Write in clear and simple language that a 6th grader could understand.
            - Include only the simplified explanation and term definitions in the output. 
            - Do not include the additional comments, or any unrelated information.
            - Provide only the explanation without repeating the input text or adding comments.

            Here is the original text to explain:
            {current_paragraph}

            Output:
            """

            # Mistral 모델에 맞게 템플릿 적용
            prompt_template = f"<s>[INST] {prompt} [/INST]"

            # 텍스트 생성
            input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
            output = model.generate(
                inputs=input_ids,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=512
            )

            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # 첫 번째 생성된 텍스트 저장
            with open(f"paragraph_{i + 1}_iteration_{iteration + 1}_generated.txt", "w", encoding="utf-8") as f:
                f.write(f"### Paragraph {i + 1}, Iteration {iteration + 1} Generated Text:\n\n{generated_text}")

            # 피드백 요청
            feedback_prompt = f"""
            Please evaluate the following explanation based on the following criteria:
            1. Does it use friendly and everyday language suitable for a 6th grader?
            2. Does it include all IT terms mentioned in the input text, with clear and relatable explanations?

            After evaluating, provide feedback on how the explanation can be improved. Write the feedback in 1-2 sentences.

            - Provide only the feedback without repeating the input text or adding comments.

            ### Explanation to Evaluate:
            {generated_text}

            Feedback:
            """

            feedback_prompt_template = f"<s>[INST] {feedback_prompt} [/INST]"

            # 피드백 생성
            input_ids_feedback = tokenizer(feedback_prompt_template, return_tensors="pt").input_ids.cuda()
            feedback_output = model.generate(
                inputs=input_ids_feedback,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=256
            )

            feedback_text = tokenizer.decode(feedback_output[0], skip_special_tokens=True)

            # 피드백 텍스트 저장
            with open(f"paragraph_{i + 1}_iteration_{iteration + 1}_feedback.txt", "w", encoding="utf-8") as f:
                f.write(f"### Paragraph {i + 1}, Iteration {iteration + 1} Feedback:\n\n{feedback_text}")

            # 피드백 반영 및 새로운 설명문 생성
            refinement_prompt = f"""
            Please revise the following explanation based on the provided feedback. Ensure that the updated explanation is clear, uses friendly language, and includes all necessary IT terms with proper definitions.

            ### Original Explanation:
            {generated_text}

            ### Feedback:
            {feedback_text}

            ### Revised Explanation Requirements:
            - Address the feedback points provided above.
            - Maintain a Flesch-Kincaid Grade Level suitable for a 6th grader.
            - Ensure clarity, conciseness, and relatability in the revised explanation.

            - Provide only the revised explanation without repeating the input text or adding comments.

            Please generate the revised explanation below:
            """

            refinement_prompt_template = f"<s>[INST] {refinement_prompt} [/INST]"

            # 피드백을 반영한 새로운 텍스트 생성
            input_ids_refined = tokenizer(refinement_prompt_template, return_tensors="pt").input_ids.cuda()
            refined_output = model.generate(
                inputs=input_ids_refined,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                max_new_tokens=512
            )

            refined_text = tokenizer.decode(refined_output[0], skip_special_tokens=True)

            # 피드백 반영 결과 저장
            with open(f"paragraph_{i + 1}_iteration_{iteration + 1}_refined.txt", "w", encoding="utf-8") as f:
                f.write(f"### Paragraph {i + 1}, Iteration {iteration + 1} Refined Text:\n\n{refined_text}")

            # 개선된 설명문을 다음 반복에 사용
            current_paragraph = refined_text

        except Exception as e:
            print(f"Error processing Paragraph {i + 1}, Iteration {iteration + 1}: {e}")
            with open("error_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Error in Paragraph {i + 1}, Iteration {iteration + 1}: {e}\n")
            continue  # 다음 문단 또는 반복으로 넘어가기

print("Processing complete. All results (and any errors) have been logged.")
