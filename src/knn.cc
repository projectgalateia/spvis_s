#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static class KNN : Classifier {
private:
	const int K = 5;

	struct Rank {
		int which;
		float score;
	};

	TrainData data;

public:
	KNN()
	{
		registerClassifier("KNN", this);
	}

	void initialize(const TrainData &data)
	{
		this->data = data;
	}

	void step()
	{
	}

	void classify(const Point &point, Likelihood &l)
	{
		std::vector<Rank> rank;

		for (const auto &d : data) {
			const float score = sqrt(pow(d.first.x - point.x, 2) + pow(d.first.y - point.y, 2));

			if (d.second.class1 > 0) {
				rank.push_back({1, score});
			} else {
				rank.push_back({0, score});
			}
		}

		std::sort(rank.begin(), rank.end(), [](const Rank &r1, const Rank &r2)
		{
			return r1.score < r2.score;
		});

		l.class1 = 0;
		l.class2 = 0;

		int end = K > rank.size() ? rank.size() : K;

		for (int i = 0; i < end; ++i) {
			if (rank[i].which) {
				l.class1 += 1;
			} else {
				l.class2 += 1;
			}
		}

		l.class1 = (l.class1 > l.class2);
		l.class2 = 1 - l.class1;
	}
} classifier;
