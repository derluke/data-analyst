# DataAnalyst

DataAnalyst is a talk-to-your-data experience. Upload a csv, ask a question, and
DataAnalyst will recommend business analyses then produce charts and tables to
answer your question (including the source code). All with ML Ops to host, monitor,
and govern the multiple components.

> [!WARNING]
> Application templates are intended to be starting points that provide guidance on how to develop, serve, and maintain AI applications.
> They require a developer or data scientist to adapt and modify them for their business requirements before being put into production.

![Using DataAnalyst](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/dataanalyst.gif)


## Table of contents
- [DataAnalyst](#dataanalyst)
  - [Table of contents](#table-of-contents)
  - [Setup](#setup)
  - [Architecture overview](#architecture-overview)
  - [Why build AI Apps with DataRobot app templates?](#why-build-ai-apps-with-datarobot-app-templates)
  - [Make changes](#make-changes)
  - [Share results](#share-results)
  - [Delete all provisioned resources](#delete-all-provisioned-resources)
  - [Setup for advanced users](#setup-for-advanced-users)
  - [Data Privacy](#data-privacy)

## Setup

> [!IMPORTANT]  
> If you are running in DataRobot Codespaces, `pulumi` is already configured and the repo automatically cloned;
> please skip to **Step 3**.
1. If `pulumi` is not already installed, install the CLI following instructions [here](https://www.pulumi.com/docs/iac/download-install/). 
   After installing for the first time, restart your terminal and run:
   ```bash
   pulumi login --local  # omit --local to use Pulumi Cloud (requires separate account)
   ```

2. Clone the template repository.

   ```bash
   git clone https://github.com/datarobot/data-analyst.git
   cd data-analyst
   ```

3. Rename the file `.env.template` to `.env` in the root directory of the repo and populate your credentials.
   This template is pre-configured to use an Azure OpenAI endpoint. If you wish to use a different LLM provider, modifications to the code will be [necessary](#change-the-llm). Note that DataAnalyst currently works with AzureOpenAI models by default and leverages `gpt-4o` and `gpt-4o-mini`: you will need access to these models for the template to work off-the-shelf.

   ```bash
   DATAROBOT_API_TOKEN=...
   DATAROBOT_ENDPOINT=...  # e.g. https://app.datarobot.com/api/v2
   OPENAI_API_KEY=...
   OPENAI_API_VERSION=...  # e.g. 2024-02-01
   OPENAI_API_BASE=...  # e.g. https://your_org.openai.azure.com/
   OPENAI_API_DEPLOYMENT_ID=...  # e.g. gpt-4
   PULUMI_CONFIG_PASSPHRASE=...  # required, choose your own alphanumeric passphrase to be used for encrypting pulumi config
   ```
   Use the following resources to locate the required credentials:
   - **DataRobot API Token**: Refer to the *Create a DataRobot API Key* section of the [DataRobot API Quickstart docs](https://docs.datarobot.com/en/docs/api/api-quickstart/index.html#create-a-datarobot-api-key).
   - **DataRobot Endpoint**: Refer to the *Retrieve the API Endpoint* section of the same [DataRobot API Quickstart docs](https://docs.datarobot.com/en/docs/api/api-quickstart/index.html#retrieve-the-api-endpoint).
   - **LLM Endpoint and API Key**: Refer to the [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=programming-language-python#retrieve-key-and-endpoint).

4. In a terminal run:
   ```bash
   python quickstart.py YOUR_PROJECT_NAME  # Windows users may have to use `py` instead of `python`
   ```
   Python 3.9+ is required.


Advanced users desiring control over virtual environment creation, dependency installation, environment variable setup
and `pulumi` invocation see [here](#setup-for-advanced-users).

## Architecture overview
TODO: Diagram here


App templates contain three families of complementary logic. For Guarded RAG you can [opt-in](#make-changes) to fully 
custom RAG logic and a fully custom frontend or utilize DR's off the shelf offerings:

- **AI logic**: Necessary to service AI requests and produce predictions and completions.
  ```
  deployment_*/  # Chat Agent Model
  ```
- **App Logic**: Necessary for user consumption; whether via a hosted front-end or integrating into an external consumption layer.
  ```
  frontend/  # Streamlit frontend
  utils/  # App business logic & runtime helpers
  ```
- **Operational Logic**: Necessary to activate DataRobot assets.
  ```
  __main__.py  # Pulumi program for configuring DataRobot to serve and monitor AI and App logic
  infra/  # Settings for resources and assets created in DataRobot
  ```

## Why build AI Apps with DataRobot app templates?

App Templates transform your AI projects from notebooks to production-ready applications. Too often, getting models into production means rewriting code, juggling credentials, and coordinating with multiple tools & teams just to make simple changes. DataRobot's composable AI apps framework eliminates these bottlenecks, letting you spend more time experimenting with your ML and app logic and less time wrestling with plumbing and deployment.

- Start building in minutes: Deploy complete AI applications instantly, then customize the AI logic or the front-end independently (no architectural rewrites needed).
- Keep working your way: Data scientists keep working in notebooks, developers in IDEs, and configs stay isolated. Update any piece without breaking others.
- Iterate with confidence: Make changes locally and deploy with confidence. Spend less time writing and troubleshooting plumbing and more time improving your app.

Each template provides an end-to-end AI architecture, from raw inputs to deployed application, while remaining highly customizable for specific business requirements.

## Make changes

TODO: Add directions to change here

## Share results

1. Log into the DataRobot application.
2. Navigate to **Registry > Applications**.
3. Navigate to the application you want to share, open the actions menu, and select **Share** from the dropdown.

## Delete all provisioned resources
```bash
pulumi down
```


## Setup for advanced users
For manual control over the setup process adapt the following steps for MacOS/Linux to your environent:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source set_env.sh
pulumi stack init YOUR_PROJECT_NAME
pulumi up 
```
e.g. for Windows/conda/cmd.exe this would be:
```bash
conda create --prefix .venv pip
conda activate .\.venv
pip install -r requirements.txt
set_env.bat
pulumi stack init YOUR_PROJECT_NAME
pulumi up 
```
For projects that will be maintained, DataRobot recommends forking the repo so upstream fixes and improvements can be merged in the future.

## Data Privacy
Your data privacy is important to us. Data handling is governed by the DataRobot [Privacy Policy](https://www.datarobot.com/privacy/), please review before using your own data with DataRobot.