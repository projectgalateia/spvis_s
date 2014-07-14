#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

static class LinearClassifier : Classifier {
private:
	float A, B, C;
	float dt;
	
	std::vector<std::pair<Point, Likelihood>> data;

	inline float calc(const Point &p)
	{
		return (A * p.x + B * p.y + C);
	}

public:
	LinearClassifier()
	{
		A = 0.0f;
		B = 0.0f;
		C = 0.0f;
		dt = 0.5f;

		//registerClassifier("Linear Classifier", this);
	}

	void initialize(const TrainData &data)
	{
		this->data.clear();
		this->data.insert(this->data.begin(), data.begin(), data.end());
	}

	void step()
	{
#pragma omp parallel for reduction(+:A, +:B, +:C)
		for (int i = 0; i < data.size(); ++i) {
			const auto &d = data[i];

			float v = calc(d.first) > 0;
			float y = d.second.class1;

			v = y - v;

			A += v * d.first.x * dt;
			B += v * d.first.y * dt;
			C += v * dt;
		}
	}

	void classify(const Point &point, Likelihood &l)
	{
		l.class1 = calc(point);
		l.class2 = -l.class1;
	}
} classifier;

