#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>

static class Perceptron : Classifier {
private:
	float A, B, C;

	const float dt = 0.1f;

	TrainData data;

	float step_function(float v)
	{
		return 1.0f / (1 + exp(-v));
	}

	void calc(const Point &p, Likelihood &l)
	{
		l.class1 = step_function(A * p.x + B * p.y + C);
		l.class2 = 1 - l.class1;
	}

public:
	Perceptron()
	{
		A = 0.1f;
		B = -0.1f;
		C = 0.0f;

		registerClassifier("Perceptron", this);
	}

	void initialize(const TrainData &data, const TrainVector &vector) 
	{
		this->data = data;
	}

	void step()
	{
		float err = 0.0f;

		for (const auto &d : data) {
			Likelihood l;

			calc(d.first, l);

			float y = d.second.class1 - d.second.class2;
			float o = l.class1 - l.class2;

			if (y * o < 0) {
				A += y * d.first.x * dt;
				B += y * d.first.y * dt;
				C += y * dt;
			}

			float v = y - o;

			err += v * v;

		}
		
		printf("err: %f\n", err);
	}

	void classify(const Point &point, Likelihood &l)
	{
		calc(point, l);
	}
} classifier;

