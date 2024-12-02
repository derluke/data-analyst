# DataAnalyst

DataAnalyst is a talk-to-your-data experience. Upload a csv, ask a question, and
DataAnalyst will recommend business analyses then produce charts and tables to
answer your question (including the source code). All with ML Ops to host, monitor,
and govern the multiple components.

![Using DataAnalyst](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/dataanalyst.gif)

DataAnalyst uses `gpt-4` (generating python code) and `gpt-4o` (summarizing charts) by default;
you will need access to these models for the template to work off-the-shelf.

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
   git clone https://github.com/datarobot-community/guarded-rag-assistant.git
   cd guarded-rag-assistant
   ```

3. Rename the file `.env.template` to `.env` in the root directory of the repo and populate your credentials.
   This template is pre-configured to use an Azure OpenAI endpoint. If you wish to use a different LLM provider, modifications to the code will be [necessary](#change-the-llm).

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

Advanced users desiring control over virtual environment creation, dependency installation, environment variable setup
and `pulumi` invocation see [here](#setup-for-advanced-users).



### Details

Instructions for installing Pulumi are [here](https://www.pulumi.com/docs/iac/download-install/). In many cases this can be done
with:

```
curl -fsSL https://get.pulumi.com | sh
```

Restart your terminal.

```
pulumi login --local

source set_env.sh
# on Windows: set_env.bat or Set-Env.ps1
```

Python must be installed for this project to run. By default, pulumi will use the Python binary aliased to `python3` to create a new virtual environment.  For projects that will be maintained, DataRobot recommends forking the repo so upstream fixes and improvements can be merged in the future.

### Feature flags

This app template requires certain feature flags to be enabled or disabled in your DataRobot account. The required feature flags can be found in [infra/feature_flag_requirements.yaml](infra/feature_flag_requirements.yaml). Contact your DataRobot representative or administrator for information on enabling the feature.


## Make Changes
### Modify the front-end

1. Ensure you have already run `pulumi up` at least once (to provision the time series deployment).
2. Streamlit assets are in `frontend/` and can be directly edited. After provisioning the stack 
   at least once, you can also test the frontend locally using `streamlit run app.py` from the
   `frontend/` directory (don't forget to initialize your environment using `source set_env.sh`).
3. Run `pulumi up` again to update your stack with the changes.


## Share results

1. Log into app.datarobot.com
2. Navigate to **Registry > Application**.
3. Navigate to the application you want to share, open the actions menu, and select **Share** from the dropdown.


## Delete all provisioned resources

```
pulumi down
```
