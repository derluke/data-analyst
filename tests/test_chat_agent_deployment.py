import datarobot as dr
import pytest
from openai import OpenAI


@pytest.fixture
def chat_client(chat_agent_deployment_id, dr_client) -> OpenAI:
    deployment_chat_base_url = (
        dr.Client().endpoint + f"/deployments/{chat_agent_deployment_id}/"
    )

    client = OpenAI(api_key=dr_client.token, base_url=deployment_chat_base_url)
    return client


def test_can_chat(chat_client: OpenAI):
    model = "gpt-4o"
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    response = chat_client.chat.completions.create(model=model, messages=messages)
    assert "paris" in response.choices[0].message.content.lower()


# @pytest.fixture(scope="class")
# def diabetes_dataset_url():
#     return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv"

# @pytest.fixture(scope="class")
# def dataset(diabetes_dataset_url):
#     df = pd.read_csv(diabetes_dataset_url)
#     # Replace non-JSON compliant values
#     df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
#     df = df.where(pd.notnull(df), None)  # Replace NaN with None

#     # Create dataset dictionary
#     dataset = {
#         "name": os.path.splitext(os.path.basename(diabetes_dataset_url))[0],
#         "data": df.to_dict("records"),
#     }
