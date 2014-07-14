#include "spvis.hpp"

#include <cstdio>
#include <cmath>

extern Classifiers classifiers;

static class LinearClassifier : Classifier {
public:
	LinearClassifier()
	{
		printf("Hello, World!\n");

		classifiers["Linear Classifier"] = this;
	}

	void initialize(const TrainData &data)
	{
	}

	void step()
	{
	}

	void classify(const Point &point, Likelihood &l)
	{
	}
} classifier;

