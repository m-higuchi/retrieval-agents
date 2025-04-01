#!/bin/bash

echo "🔄 Waiting for Elasticsearch to be ready..."
until curl -s -u elastic:${ELASTIC_PASSWORD} http://elasticsearch:9200 >/dev/null; do
  sleep 2
done

echo "✅ Elasticsearch is up. Creating Kibana service token..."
if [ ! -f config/service_tokens/kibana/kibana-token ]; then
  bin/elasticsearch-service-tokens create kibana kibana-token
fi

TOKEN=$(cat config/service_tokens/kibana/kibana-token)
echo "🔐 Token generated: $TOKEN"

echo "ELASTICSEARCH_SERVICEACCOUNTTOKEN=$TOKEN" > ./kibana.env
echo "✅ Token saved to kibana.env"
