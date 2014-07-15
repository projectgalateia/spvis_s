#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

static class VoidClassifier : Classifier {
private:

public:
	VoidClassifier()
	{
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
