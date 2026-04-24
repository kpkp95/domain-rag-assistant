mkdir -p src
mkdir -p research
mkdir -p .github/workflows
mkdir -p data/medical
mkdir -p data/machine_learning
mkdir -p data/llm
mkdir -p static
mkdir -p templates
mkdir -p evaluation

# Creating files
touch src/__init__.py
touch src/helper.py
touch src/prompt.py

touch .env
touch setup.py
touch requirements.txt
touch Dockerfile
touch app.py
touch store_index.py
touch .gitignore

touch research/trials.ipynb

touch static/style.css
touch static/script.js
touch templates/chat.html

touch evaluation/questions.json
touch evaluation/evaluate.py
touch evaluation/results.csv

touch .github/workflows/main.yml

echo "Directory structure created successfully!"