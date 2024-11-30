#include "graph.hpp"

void loadTrainingDataFromJson(const string& filename, vector<pair<string, string>>& trainingData, vector<pair<string, string>>& validationData) {
	ifstream file(filename);
	if (file.is_open()) {
		json j;
		file >> j;

		for (const auto& item : j["training_data"]) {
			string word = item["word"];
			string language = item["language"];
			trainingData.emplace_back(word, language);
		}

		for (const auto& item : j["validation_data"]) {
			string word = item["word"];
			string language = item["language"];
			validationData.emplace_back(word, language);
		}
	}
	else {
		throw runtime_error("Unable to open training data file.");
	}
}

void trainWithDynamicEdges(Graph& g, const vector<pair<string, string>>& trainingData, double learningRate) {
	for (const auto& example : trainingData) {
		const string& word = example.first;
		const string& trueLanguage = example.second;

		g.addNode(word);
		g.addNode(trueLanguage);

		if (!g.edgeExists(word, trueLanguage)) {
			g.addEdge(word, trueLanguage, 0.0);
		}

		for (const auto& neighbor : g.getNeighbors(word)) {
			const string& language = neighbor.first;
			if (language == trueLanguage) {
				g.updateWeight(word, language, learningRate);
			}
			else {
				g.updateWeight(word, language, -learningRate);
			}
		}
	}
}

double adjustLearningRate(double initialRate, int epoch, double decayRate) {
	return initialRate / (1 + decayRate * epoch);
}

bool earlyStopping(const Graph& g, const vector<pair<string, string>>& validationData, double& bestAccuracy, int& noImprovementCount, int patience) {
	double correct = 0;
	for (const auto& example : validationData) {
		if (g.classify(example.first) == example.second) {
			correct++;
		}
	}
	double accuracy = correct / validationData.size();
	if (accuracy > bestAccuracy) {
		bestAccuracy = accuracy;
		noImprovementCount = 0;
		return false;
	}
	else {
		noImprovementCount++;
		return noImprovementCount >= patience;
	}
}

void normalizeWeights(Graph& g) {
	for (const auto& node : g.getNodes()) {
		double totalWeight = 0.0;

		for (const auto& neighbor : g.getNeighbors(node.first)) {
			totalWeight += neighbor.second;
		}

		for (auto& neighbor : g.getNeighbors(node.first)) {
			g.updateWeight(node.first, neighbor.first, neighbor.second / totalWeight - neighbor.second);
		}
	}
}

void trainModel(Graph& g, const vector<pair<string, string>>& trainingData, const vector<pair<string, string>>& validationData,
	double initialRate, int epochs, double decayRate, int patience) {
	double bestAccuracy = 0.0;
	int noImprovementCount = 0;

	for (int epoch = 0; epoch < epochs; ++epoch) {
		double learningRate = adjustLearningRate(initialRate, epoch, decayRate);
		trainWithDynamicEdges(g, trainingData, learningRate);
		normalizeWeights(g);

		if (earlyStopping(g, validationData, bestAccuracy, noImprovementCount, patience)) {
			cout << "Training stopped early at epoch " << epoch + 1 << endl;
			break;
		}

		cout << "Epoch " << epoch + 1 << ": Best accuracy = " << bestAccuracy << endl;
	}
}

int main() {
	Graph g;
	vector<pair<string, string>> trainingData;
	vector<pair<string, string>> validationData;

	string trainingFile = "training_data.json";
	string similarityFile = "word_similarity.json";

	try {
		loadTrainingDataFromJson(trainingFile, trainingData, validationData);
		g.loadWordSimilarities(similarityFile);
	}
	catch (const exception& ex) {
		cerr << "Error: " << ex.what() << endl;
		return 1;
	}

	double initialRate = 0.1;
	double decayRate = 0.01;
	int epochs = 10;
	int patience = 3;

	trainModel(g, trainingData, validationData, initialRate, epochs, decayRate, patience);

	cout << "Predicted language for 'boat': " << g.classify("boat") << endl;
	cout << "Similarity between 'car' and 'automobile': " << g.getWordSimilarity("car", "automobile") << endl;

	return 0;
}
