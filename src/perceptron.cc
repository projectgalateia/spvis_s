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

	void initialize(const TrainData &data)
	{
		this->data = data;
	}

	void step()
	{
		float err = 0.0f;

		for (const auto &d : data) {
			Likelihood l;

			calc(d.first, l);

			float v = 0.0f;

			if (d.second.class1 > 0) {
				v = 1 - l.class1;
			} else {
				v = 0 - l.class1;
			}

			err += v * v;

			A += v * d.first.x * dt;
			B += v * d.first.y * dt;
			C += v * dt;
		}
		
		printf("err: %f\n", err);
	}

	void classify(const Point &point, Likelihood &l)
	{
		calc(point, l);
	}
} classifier;
