from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input long text
text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on
creating intelligent machines capable of mimicking human cognitive functions such as learning,
problem-solving, and decision-making.

In recent years, AI has significantly impacted various industries, including healthcare, finance,
education, and entertainment. AI-powered applications, such as chatbots, self-driving cars,
and recommendation systems, have transformed the way we interact with technology. Machine learning
and deep learning, subsets of AI, enable systems to learn from data and improve over time without
explicit programming.

However, AI also poses ethical challenges, such as bias in decision-making and concerns over job displacement.
As AI technology continues to advance, it is crucial to balance innovation with ethical considerations
to ensure its responsible development and deployment.
"""

# Define summarization strategies
strategies = {
    "default": {"do_sample": False},
    "high_randomness": {"do_sample": True, "temperature": 0.9},
    "conservative": {"do_sample": False, "num_beams": 5},
    "diverse": {"do_sample": True, "top_k": 50, "top_p": 0.95}
}

# Generate summaries using different strategies
def summarize_text(text, strategy):
    summary = summarizer(text, max_length=130, min_length=30, **strategies[strategy])[0]['summary_text']
    return summary

# Summarize and print using multiple strategies
print("Original Text:\n", text)
for strategy_name in strategies:
    print(f"\n{strategy_name.capitalize()} Summary:\n", summarize_text(text, strategy_name))
