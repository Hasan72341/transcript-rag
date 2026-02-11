create-zip:
	zip -r transcript-rag.zip transcript-rag/ queries-generation/ images/ simulated-real-query-system/ technical-report/ docker-compose.yml README.md .gitignore
	
upload-zip:
	gsutil -m cp transcript-rag.zip gs://recor-rag