# User Manual

Welcome to the NOCTUA system -- a research assistant that combines the
power of vector search, retrieval-augmented generation (RAG), and
multi-step query analysis.

This manual is intended for both new users unfamiliar with the platform
and experienced researchers seeking to make the most of the system's
capabilities.

The system allows you to:

-   search for relevant information in a scientific abstracts database
    > (arXiv.org);

-   analyze one or more PDF documents;

-   ask questions about internal documentation of the platform itself;

-   use a query refinement mechanism to improve results.

All components of the system are presented as interactive pages. This
manual explains in detail the purpose of each page, available user
actions, and expected outcomes.

Below you will find a description of the navigation menu, operational
modes, tips on how to formulate queries, and examples of interaction
with the system.

## 1. Welcome Page and Navigation

When the application starts, the welcome page appears, introducing users
to the capabilities of the NOCTUA system. It includes:

-   A heading and a brief description of the system\'s purpose;

-   A list of the platform's main features, each accompanied by an icon;

-   An explanation of pipeline structure (working stages: retrieval,
    generation, refinement, etc.);

-   Tips on how to get started: how to select a mode, enable query
    refinement, or upload a PDF.

This page does not perform any analytical functions --- it is purely
introductory and informative.

On the left side, there is a navigation menu available on all pages. It
allows quick switching between modes.

## 2 Abstract Search Page

This section allows users to obtain a concise overview of scientific
publications related to a topic of interest. Simply enter your query
into the text field -- the system will retrieve the most relevant
articles, generate a summary report, and suggest ideas for further
research.

Optionally, you can enable the refinement mode to expand or adjust your
query after the initial analysis. This feature allows for more accurate
answers tailored to your interests.

In addition to the standard navigation menu, the sidebar contains an
Info and Settings section that provides a quick diagnostic of the
system\'s connection to external services.

MongoDB Status indicates whether the connection to the scientific
metadata database was successful. If connected, it also shows the number
of documents loaded (e.g., Connected • docs loaded: 20000).

OpenAI API Status reports the availability of language models used for
answer generation and vectorization. A message such as Connected is
shown upon successful connection.

These statuses are updated automatically every time the page is opened
and require no user action. If a connection is unavailable, an error
message is displayed in red instead of the green indicator.

Below the status indicators is the Enable refinement mode button, which
toggles query refinement.

### 2.1 Working Without Query Refinement

To initiate a search, enter a query in the text field at the top of the
page and click the Run Query button. The system will automatically
analyze the available abstracts and generate a report.

While the query is being processed, progress indicators are shown on
screen, reflecting the sequence of steps: retrieving relevant articles,
generating the answer, and preparing follow-up ideas.

Once the generation process is complete, the user is presented with:

-   a structured answer based on the content of the retrieved abstracts;

-   a block of research ideas, displayed below the main text.

The generation of research ideas is the third, but equally important,
step. The system automatically proposes directions to help the user
refine or expand their topic. Even if few relevant articles are found or
the response is too generic, these suggestions offer additional depth.

The proposed ideas are based on both the original query and the themes
of the retrieved publications. They may inspire interdisciplinary
approaches or help formulate new, more specific questions. Typically,
three suggestions are provided, focused on practical relevance and
scientific novelty.

### 2.2 Working with Query Refinement Enabled

The query refinement mode allows the system to produce more accurate,
detailed answers tailored to your specific information needs.

To activate it, simply check the Enable refinement mode option in the
sidebar before launching the initial query. Then, enter your question in
the query field and click Run Query, just like in the basic mode.

After the initial response is generated, an additional block appears --
Clarification Questions. This block contains follow-up questions
automatically generated by the language model based on the first
response.

The user can read these questions, reflect on what might be missing in
the answer, and provide additional context or clarification in a new
text field.

When the Get Refined Answer button is pressed, the system reruns the
query, combining the original and refined inputs, and produces an
updated response that incorporates the extended context.

As before, a new set of research suggestions appears below the refined
response. This flexible approach is particularly useful for deeper
exploration of complex or interdisciplinary topics.

## 3 Single PDF Mode Page

This page enables detailed analysis of a specific PDF document. You can
upload any scientific publication in PDF format, and the system will
automatically perform all necessary steps: store it in MongoDB, convert
it to Markdown, split it into chunks, generate vector embeddings, and
update the vector index. After processing, you can ask any question
about the document's content and receive an answer based on the most
relevant text segments.

The sidebar displays connection statuses for the database and OpenAI
API. In addition, the Document Upload section provides a field for
uploading a PDF file. Once a file is selected, it is stored temporarily,
and the Process PDF button becomes available to start the analysis.

After successful upload and processing of the PDF file, you can ask
questions about its content and click Run Query to initiate the
generation.

The system performs a multi-step reasoning process known as
Chain-of-Thought.

During this process:

-   The system generates a list of clarification questions based on your
    initial query;

-   Each question is answered in turn, refining key aspects of the
    topic;

-   Once all steps are complete, the answers are combined and analyzed;

-   Based on the intermediate reasoning, the system produces a final,
    comprehensive answer that best reflects the essence of the original
    query.

Each reasoning step is displayed separately (e.g., Step 1, Step 2,
etc.), allowing you to trace the logic of the model's thinking.

This approach helps the model better understand the document's content,
reduces the likelihood of incorrect conclusions, and makes the
generation process more transparent to the user.

The final answer is displayed at the end in a dedicated block labeled
Final Answer.

## 4 Multi-PDF Mode Page

This page works similarly to the Single PDF Mode described in section
4.3 but allows users to analyze multiple PDF documents simultaneously.

In Multi-PDF Mode, the Chain-of-Thought process is modified to take into
account information from multiple sources.

This mode is a convenient tool for simultaneously reviewing multiple
publications and identifying their similarities or differences.

## 5 Help & Guide Page

This mode allows users to ask open-ended questions about system
functionality, which are answered based on the uploaded user manual. The
interface is designed for intuitive navigation, manual browsing, and
interactive assistance.

Users can view the full manual by clicking the Show Manual button, which
opens an embedded Markdown viewer displaying the structure and contents
of all system features. By entering a question and clicking Ask, users
receive a response based on the relevant section of the manual.

The sidebar shows the status of manual ingestion into the vector
database. A Download Manual button is also available, allowing users to
save the current version of the guide in Markdown format to their
device.
