import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import requests
from bs4 import BeautifulSoup
import clickhouse_driver

def fetch_text_from_document(url):
    response = requests.get(url)  # Отправляем запрос по переданной ссылке
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')  # Парсим HTML страницу
        text = ""
        for paragraph in soup.find_all('p'):  # Извлекаем текст из тегов <p>
            text += paragraph.get_text() + " "
        return text
    else:
        print("Error: Failed to fetch document text")
        return None

# Load and preprocess the data
with open('docs_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

documents = []
for doc in data:
    # Fetch the text from the document using API/Scraper based on the document link
    text = fetch_text_from_document(doc['ссылка на документ'])
    documents.append(text)

# Initialize the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Encode the documents
encoded_docs = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

# Generate embeddings for the documents
with torch.no_grad():
    document_embeddings = model.encoder(
        input_ids=encoded_docs['input_ids'],
        attention_mask=encoded_docs['attention_mask']
    ).last_hidden_state

clickhouse_connection = clickhouse_driver.connect('localhost')
clickhouse_cursor = clickhouse_connection.cursor()

document_embeddings_list = document_embeddings.squeeze().tolist()

for idx, embedding in enumerate(document_embeddings_list):
    query = f"INSERT INTO document_embeddings (document_id, embedding) VALUES ({idx}, {embedding})"
    clickhouse_cursor.execute(query)

clickhouse_connection.close()

query = "Какой размер процентной ставки по депозитным операциям Центробанка?"

# Получение эмбеддинга запроса
encoded_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    query_embedding = model.encoder(
        input_ids=encoded_query['input_ids'],
        attention_mask=encoded_query['attention_mask']
    ).last_hidden_state.mean(dim=1)

# Сравнение запроса с документами для поиска наилучшего документа
similarity_scores = torch.cosine_similarity(query_embedding, document_embeddings, dim=1)
best_document_idx = torch.argmax(similarity_scores).item()
best_document = documents[best_document_idx]

# Генерация ответа на запрос
inputs = tokenizer("answer: " + query + " documents: " + best_document, return_tensors='pt', padding=True, truncation=True)
generated = model.generate(**inputs)

# Расшифровка сгенерированного ответа
decoded_answer = tokenizer.decode(generated[0], skip_special_tokens=True)
print(decoded_answer)