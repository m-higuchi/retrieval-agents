# Retrieval Agents

[![CI](https://github.com/m-higuchi/retrieval-agents/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/m-higuchi/retrieval-agents/actions/workflows/unit-tests.yml)

## 概要

このプロジェクトは、Retrieval-Augmented Generation (RAG) を活用した大規模言語モデル (LLM) のアプリケーション開発を目指す。RAGは、外部データソースを利用して生成するコンテンツの精度を高める技術であり、本プロジェクトでは、ユーザーの入力に基づいて関連する情報を検索し、その情報をもとにLLMが精緻な応答を生成するシステムを構築する。

## 主な機能

- **外部データソースとの連携:** Mongo DB、Chroma DB等のベクトルDBやOpenAI、AnthropicなどのLLM APIと連携し、外部の知識を活用した高度な情報検索機能を実装する。

- **情報検索と生成の組み合わせ:** ユーザーからの質問に対して、まず関連するデータを検索し、そのデータを基にLLMが回答を生成する。これにより、従来のLLMだけでは難しい特定の情報を正確に提供することが可能になる。


RAGの利点としては、次の点が挙げられる:

- **情報の多様性:** 複数の情報源からデータを取得するため、LLM単独での学習データだけでは得られない多様な知識を組み込むことができる。

- **精度の向上:** ユーザーの質問に対して、関連する実際のデータを用いることで、より高精度でコンテキストに即した回答を生成できる。

今後のアップデートにおいては、RAGを用いたより高度な機能の追加や、さらに多様なデータソースとの統合を予定している。

## 前提条件

- VS Code
- Docker

インストールされていない場合は[Installation Guide](./docs/INSTALL.md)を参照。

### インストール

1. リポジトリのクローン
    ```
    git clone git@github.com:m-higuchi/retrieval-agents.git
    cd retrieval-agents
    ```


> [!CAUTION]
> リポジトリは将来的にはGitHub組織アカウントのリポジトリに移行予定。移行後はリポジトリURLが変更されるため要注意。

> [!IMPORTANT]
> このプロジェクトはDev Container環境を使用することを推奨する。特別な理由がない限り、以降の作業は[Dev Container Setup](./docs/DEV_CONTAINER_SETUP.md)に従ってDev Containerを開き、コンテナ内で作業を行う。


2. `.env`ファイルを作成する。

    ```bash
    cp .env.example .env
    ```

3. 使用するAPIプロバイダーのAPIキー等を設定し`.env` ファイルに保存する。
    ```
    OPENAI_API_KEY=sk-...
    ```

> [!CAUTION]
> 現時点ではOpen AI, Anthropic, Nomic AI, Chroma DBのみ動作確認済み。

4. アプリケーションおよび依存関係パッケージをインストールする。Dev Containerを使用している場合、この手順はコンテナ起動時に自動実行されるため不要。

    ```
    poetry install --with dev --all-extras
    ```

### デバッグと実行

`langgraph dev --allow-blocking`コマンドを実行すると開発モードでLangGraph APIサーバーがローカルホストで起動する。

```
langgraph dev --allow-blocking
```

> [!NOTE]
> 現状では`--allow-blocking`オプションが無いとエラーが出てしまう。同期的なI/O操作が原因であると思われるが解消できていない。

正常に起動すると以下のように出力される。

```
INFO:langgraph_api.cli:

        Welcome to

╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

This in-memory server is designed for development and testing.
For production use, please use LangGraph Cloud.
```

以下のいずれかの方法で実行可能。

- **Web UI:** [Studio UI](http://127.0.0.1:2024)にアクセスして、インタラクティブに操作を行う。
- **API:** [APIベースURL](http://127.0.0.1:2024)にアクセスして直接APIを利用する。APIの仕様は以下のAPD Docsを参照。
- **API Docs:** [APIリファレンス](http://127.0.0.1:2024/docs)。APIエンドポイントを直接試すことができる機能も提供している。

> [!WARNING]
> 以降のセクションは後日追加予定。

### Setup Retriever

The defaults values for `retriever_provider` are shown below:

```yaml
retriever_provider: elastic
```

Follow the instructions below to get set up, or pick one of the additional options.

#### Elasticsearch

Elasticsearch (as provided by Elastic) is an open source distributed search and analytics engine, scalable data store and vector database optimized for speed and relevance on production-scale workloads.

##### Setup Elasticsearch
Elasticsearch can be configured as the knowledge base provider for a retrieval agent by being deployed on Elastic Cloud (either as a hosted deployment or serverless project) or on your local environment.

**Elasticsearch Serverless**

1. Signup for a free 14 day trial with [Elasticsearch Serverless](https://cloud.elastic.co/registration?onboarding_token=search&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=langchain).
2. Get the Elasticsearch URL, found on home under "Copy your connection details".
3. Create an API key found on home under "API Key".
4. Copy the URL and API key to your `.env` file created above:

```
ELASTICSEARCH_URL=<ES_URL>
ELASTICSEARCH_API_KEY=<API_KEY>
```

**Elastic Cloud**

1. Signup for a free 14 day trial with [Elastic Cloud](https://cloud.elastic.co/registration?onboarding_token=search&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=langchain).
2. Get the Elasticsearch URL, found under Applications of your deployment.
3. Create an API key. See the [official elastic documentation](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key) for more information.
4. Copy the URL and API key to your `.env` file created above:

```
ELASTICSEARCH_URL=<ES_URL>
ELASTICSEARCH_API_KEY=<API_KEY>
```
**Local Elasticsearch (Docker)**

```
docker run -p 127.0.0.1:9200:9200 -d --name elasticsearch --network elastic-net   -e ELASTIC_PASSWORD=changeme   -e "discovery.type=single-node"   -e "xpack.security.http.ssl.enabled=false"   -e "xpack.license.self_generated.type=trial"   docker.elastic.co/elasticsearch/elasticsearch:8.15.1
```

See the [official Elastic documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html) for more information on running it locally.

Then populate the following in your `.env` file:

```
# As both Elasticsearch and LangGraph Studio runs in Docker, we need to use host.docker.internal to access.

ELASTICSEARCH_URL=http://host.docker.internal:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=changeme
```
#### MongoDB Atlas

MongoDB Atlas is a fully-managed cloud database that includes vector search capabilities for AI-powered applications.

1. Create a free Atlas cluster:
- Go to the [MongoDB Atlas website](https://www.mongodb.com/cloud/atlas/register) and sign up for a free account.
- After logging in, create a free cluster by following the on-screen instructions.

2. Create a vector search index
- Follow the instructions at [the Mongo docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/)
- By default, we use the collection `langgraph_retrieval_agent.default` - create the index there
- Add an indexed filter for path `user_id`
- **IMPORTANT**: select Atlas Vector Search NOT Atlas Search when creating the index
Your final JSON editor configuration should look something like the following:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "user_id",
      "type": "filter"
    }
  ]
}
```

The exact numDimensions may differ if you select a different embedding model.

2. Set up your environment:
- In the Atlas dashboard, click on "Connect" for your cluster.
- Choose "Connect your application" and copy the provided connection string.
- Create a `.env` file in your project root if you haven't already.
- Add your MongoDB Atlas connection string to the `.env` file:

```
MONGODB_URI="mongodb+srv://username:password@your-cluster-url.mongodb.net/?retryWrites=true&w=majority&appName=your-cluster-name"
```

Replace `username`, `password`, `your-cluster-url`, and `your-cluster-name` with your actual credentials and cluster information.
#### Pinecone Serverless

Pinecone is a managed, cloud-native vector database that provides long-term memory for high-performance AI applications.

1. Sign up for a Pinecone account at [https://login.pinecone.io/login](https://login.pinecone.io/login) if you haven't already.

2. After logging in, generate an API key from the Pinecone console.

3. Create a serverless index:
   - Choose a name for your index (e.g., "example-index")
   - Set the dimension based on your embedding model (e.g., 1536 for OpenAI embeddings)
   - Select "cosine" as the metric
   - Choose "Serverless" as the index type
   - Select your preferred cloud provider and region (e.g., AWS us-east-1)

4. Once you have created your index and obtained your API key, add them to your `.env` file:

```
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=your-index-name
```


### Setup Model

The defaults values for `response_model`, `query_model` are shown below:

```yaml
response_model: anthropic/claude-3-5-sonnet-20240620
query_model: anthropic/claude-3-haiku-20240307
```

Follow the instructions below to get set up, or pick one of the additional options.

#### Anthropic

To use Anthropic's chat models:

1. Sign up for an [Anthropic API key](https://console.anthropic.com/) if you haven't already.
2. Once you have your API key, add it to your `.env` file:

```
ANTHROPIC_API_KEY=your-api-key
```
#### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```



### Setup Embedding Model

The defaults values for `embedding_model` are shown below:

```yaml
embedding_model: openai/text-embedding-3-small
```

Follow the instructions below to get set up, or pick one of the additional options.

#### OpenAI

To use OpenAI's embeddings:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```

#### Cohere

To use Cohere's embeddings:

1. Sign up for a [Cohere API key](https://dashboard.cohere.com/welcome/register).
2. Once you have your API key, add it to your `.env` file:

```bash
COHERE_API_KEY=your-api-key
```





<!--
End setup instructions
-->

## Using

```bash
```
Once you've set up your retriever saved your model secrets, it's time to try it out! First, let's add some information to the index. Open studio, select the "indexer" graph from the dropdown in the top-left, provide an example user ID in the configuration at the bottom, and then add some content to chat over.

```json
[{ "page_content": "My cat knows python." }]
```

When you upload content, it will be indexed under the configured user ID. You know it's complete when the indexer "delete"'s the content from its graph memory (since it's been persisted in your configured storage provider).

Next, open the "retrieval_graph" using the dropdown in the top-left. Ask it about your cat to confirm it can fetch the required information! If you change the `user_id` at any time, notice how it no longer has access to your information. The graph is doing simple filtering of content so you only access the information under the provided ID.

## Development

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls tools, updating the default system message in `src/retrieval_agents/agents/utils.py` to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://github.com/langchain-ai/langgraph) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.