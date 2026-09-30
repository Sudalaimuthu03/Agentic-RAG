# V4 Testing

The automated suite covers semantic resolution, ambiguity, variant identity, context continuation, evidence alignment, execution budgets, and answer validation.

Release validation additionally checks source-tree cleanliness, stale V3 module removal, Python syntax, internal imports, ZIP cleanliness, and clean-extraction validation.

Live Ollama/embedding/reranker tests require the user's local model environment. They must be run on the target Windows/Python 3.11 machine before declaring a production release.
