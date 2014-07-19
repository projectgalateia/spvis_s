#include "spvis.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cfloat>

static class SVM : Classifier {
private:
	const float dt = 0.1f;

	TrainVector vector;
	std::vector<float> alphas;

	struct FPoint {
		float x;
		float y;
	};

	float B;

	float screen_w;
	float screen_h;

	FPoint map(const Point &p)
	{
		FPoint ret = {1.0f - 2.0f * p.x / screen_w, 1.0f - 2.0f * p.y / screen_h};

		return ret;
	}

	float classof(const Likelihood &l)
	{
		return (l.class1 - l.class2);
	}

	float dot(const FPoint &p1, const FPoint &p2)
	{
		return (p1.x * p2.x + p1.y * p2.y);
	}

	float kernel(const Point &p1, const Point &p2)
	{
		return pow(dot(map(p1), map(p2)) + 1, 10);
	}

	void calc(const Point &p, Likelihood &l)
	{
		l.class1 = B;

		for (int i = 0; i < alphas.size(); ++i) {
			l.class1 += alphas[i] * classof(vector[i].second) * kernel(vector[i].first, p);
		}

		l.class2 = - l.class1;
	}

public:
	SVM()
	{
		registerClassifier("SVM", this);
	}
	
	void initialize(const TrainData &data, const TrainVector &vector) 
	{
		int w, h;

		screen_prop(w, h);

		screen_w = w;
		screen_h = h;

		B = 0.0f;
		this->vector = vector;

		alphas.clear();
		alphas.resize(this->vector.size());
	}

	void step()
	{
		float nB = 0.0f;

		for (int i = 0; i < vector.size(); ++i) {
			const auto &d = vector[i];

			Likelihood l;

			calc(d.first, l);

			float y = classof(d.second);
			float o = l.class1;

			float k = y * o;

			if (k <= 0) {
				alphas[i] += dt;
			}

			nB += (o - B) - y;
		}

		B = nB / vector.size();
	}

	void classify(const Point &point, Likelihood &l)
	{
		calc(point, l);
	}
} classifier;

