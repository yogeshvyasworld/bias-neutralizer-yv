import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import trafilatura
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app) # This enables chrome extension to talk to the server

#loading the API from Render's Env variables
GROQ_KEY = os.environ.get("GROQ_API_KEY")
yv_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_KEY)


def run_bias_crew(article_text):
    
    # Define the Agent
    auditor = Agent(
        role="Senior Bias Detection Auditor",
        goal="Identify all instances of emotional manipulation and biased framing.",
        backstory="You are a linguistic expert trained to spot hidden agendas and loaded language.",
        llm=yv_llm,
        verbose=True
    )

    editor = Agent(
        role="Neutral Style News Editor",
        goal="Rewrite news articles to be 100% neutral and factual.",
        backstory="You have 20 years of experience at a major news agency. You value clarity and neutrality.",
        llm=yv_llm,
        verbose=True
    )

    # Define the Task
    task1 = Task(
        description=f"Analyze this text for bias and list specific issues: {content[:4000]}",
        expected_output="A bulleted list of biased phrases and emotional triggers found.",
        agent=auditor
    )

    task2 = Task(
        description="Using the Bias Auditor's list, rewrite the original article in neutral agency Style.",
        expected_output="A clean, factual news report with no emotional language.",
        agent=editor
    )

    # Create the Crew
    yogesh_crew = Crew(
        agents=[auditor, editor],
        tasks=[task1, task2],
        process=Process.sequential
    )

    # Start the Crew
    result = yogesh_crew.kickoff()
    return result
    pass

@app.route('/process', methods=['POST'])
def process():
    url = request.json.get('url')
    content = trafilatura.extract(trafilatura.fetch_url(url))

    if not content:
        return jsonify({"error": "Could not read article"}), 400

    result = run_bias_crew(content)
    return jsonify({"cleaned_text": str(result)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)



