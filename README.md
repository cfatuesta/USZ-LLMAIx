1. install ollama: `brew install ollama`
2. start ollama (and restart on login): `brew services start ollama`
3. download llama3.2: `ollama pull llama3.2`
4. install python dependencies: `pip install ollama pandas pydantic`
5. run the code: `python process_csvs.py`