#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include "json.hpp"
#include <fstream>
#include <sstream>

using namespace std;
using json = nlohmann::json;

class Graph {
	unordered_map<string, unordered_map<string, double>> adjacencyList;
	unordered_map<string, unordered_map<string, double>> wordSimilarityList;

public:
	void addNode(const string& node) {
		if (adjacencyList.find(node) == adjacencyList.end()) {
			adjacencyList[node] = {};
		}
	}

	void addEdge(const string& node1, const string& node2, double weight) {
		addNode(node1);
		addNode(node2);
		adjacencyList[node1][node2] = weight;
	}

	void updateWeight(const string& node1, const string& node2, double delta) {
		if (adjacencyList.find(node1) != adjacencyList.end() && adjacencyList[node1].find(node2) != adjacencyList[node1].end()) {
			adjacencyList[node1][node2] += delta;
		}
	}

	vector<pair<string, double>> getNeighbors(const string& node) const {
		vector<pair<string, double>> neighbors;
		if (adjacencyList.find(node) != adjacencyList.end()) {
			for (const auto& pair : adjacencyList.at(node)) {
				neighbors.emplace_back(pair.first, pair.second);
			}
		}
		return neighbors;
	}

	const unordered_map<string, unordered_map<string, double>>& getNodes() const {
		return adjacencyList;
	}

	string classify(const string& word) const {
		double maxWeight = numeric_limits<double>::lowest();
		string bestLanguage;
		if (adjacencyList.find(word) != adjacencyList.end()) {
			for (const auto& pair : adjacencyList.at(word)) {
				if (pair.second > maxWeight) {
					maxWeight = pair.second;
					bestLanguage = pair.first;
				}
			}
		}
		return bestLanguage;
	}

	void printGraph() const {
		for (const auto& node : adjacencyList) {
			cout << node.first << ": ";
			for (const auto& neighbor : node.second) {
				cout << "(" << neighbor.first << ", " << neighbor.second << ") ";
			}
			cout << endl;
		}
	}

	bool edgeExists(const string& node1, const string& node2) const {
		return adjacencyList.find(node1) != adjacencyList.end() && adjacencyList.at(node1).find(node2) != adjacencyList.at(node1).end();
	}

	void addWordSimilarity(const string& word1, const string& word2, double similarity) {
		addNode(word1);
		addNode(word2);
		wordSimilarityList[word1][word2] = similarity;
		wordSimilarityList[word2][word1] = similarity;
	}

	double getWordSimilarity(const string& word1, const string& word2) const {
		if (wordSimilarityList.find(word1) != wordSimilarityList.end() && wordSimilarityList.at(word1).find(word2) != wordSimilarityList.at(word1).end()) {
			return wordSimilarityList.at(word1).at(word2);
		}
		return 0.0;
	}

	void loadWordSimilarities(const string& filename) {
		ifstream file(filename);
		if (file.is_open()) {
			json j;
			file >> j;
			for (auto& word : j.items()) {
				const string& word1 = word.key();
				for (auto& similarity : word.value().items()) {
					const string& word2 = similarity.key();
					double simValue = similarity.value();
					addWordSimilarity(word1, word2, simValue);
				}
			}
		}
		else {
			throw runtime_error("Unable to open similarity file.");
		}
	}
};
