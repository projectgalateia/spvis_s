#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

static class LinearClassifier : Classifier {
private:
	float A, B, C;
	float dt;
	
	TrainVector data;

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
		dt = 0.1f;

		registerClassifier("Linear Classifier", this);
	}

	void initialize(const TrainData &data)
	{
		dataToVector(data, this->data);
	}

	void step()
	{
		float dA, dB, dC;

		for (int i = 0; i < data.size(); ++i) {
			const auto &d = data[i];

			float v = calc(d.first) > 0;
			float y = d.second.class1;

			v = y - v;

			dA += v * d.first.x * dt;
			dB += v * d.first.y * dt;
			dC += v * dt;
		}

		A += dA;
		B += dB;
		C += dC;
	}

	void classify(const Point &point, Likelihood &l)
	{
		l.class1 = calc(point);
		l.class2 = -l.class1;
	}
} classifier;

