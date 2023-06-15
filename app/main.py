from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import pipeline

app = FastAPI()


class inputArticle(BaseModel):
    text: str

@app.post('/summarize')
def summarize(input: inputArticle):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    article = [input.text]  # Wrap the input text in a list
    response = summarizer(article, max_length=200, min_length=30, do_sample=False)  # Pass the list as input
    return response[0]['summary_text']  # Return the summary text

#if __name__ == "__main__":
 #   uvicorn.run("main:app", host="0.0.0.0", port=5000)