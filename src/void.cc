#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

static class VoidClassifier : Classifier {
private:

public:
	void initialize(const TrainData &data, const TrainVector &vector) 
	{
	}

	void step()
	{
	}

	void classify(const Point &point, Likelihood &l)
	{
	}
} classifier;
